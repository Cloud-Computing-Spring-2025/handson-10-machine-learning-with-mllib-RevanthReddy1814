from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
)
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    df = df.na.fill({'TotalCharges': 0})

    indexers = [
     StringIndexer(inputCol=column, outputCol=column + "_Index")
        for column in ["gender", "PhoneService", "InternetService", "Churn"]
    ]

    for indexer in indexers:
        df = indexer.fit(df).transform(df)

    encoder = OneHotEncoder(
        inputCols=["gender_Index", "PhoneService_Index", "InternetService_Index"],
        outputCols=["gender_OHE", "PhoneService_OHE", "InternetService_OHE"]
    )

    df = encoder.fit(df).transform(df)

    feature_cols = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "gender_OHE", "PhoneService_OHE", "InternetService_OHE"
    ]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df).withColumnRenamed("Churn_Index", "label")

    return df

# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train)

    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)
    print(f"Logistic Regression AUC: {auc:.4f}")

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    selector = ChiSqSelector(
        numTopFeatures=5,
        featuresCol="features",
        labelCol="label",
        outputCol="selectedFeatures"
    )
    result = selector.fit(df).transform(df)
    result.select("selectedFeatures", "label").show(5, truncate=False)

# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator()

    models = {
        "LogisticRegression": LogisticRegression(labelCol="label", featuresCol="features"),
        "DecisionTree": DecisionTreeClassifier(labelCol="label", featuresCol="features"),
        "RandomForest": RandomForestClassifier(labelCol="label", featuresCol="features"),
        "GBT": GBTClassifier(labelCol="label", featuresCol="features")
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder()
            .addGrid(models["LogisticRegression"].regParam, [0.01, 0.1])
            .addGrid(models["LogisticRegression"].elasticNetParam, [0.0, 0.5])
            .build(),

        "DecisionTree": ParamGridBuilder()
            .addGrid(models["DecisionTree"].maxDepth, [3, 5, 7])
            .build(),

        "RandomForest": ParamGridBuilder()
            .addGrid(models["RandomForest"].numTrees, [10, 20])
            .build(),

        "GBT": ParamGridBuilder()
            .addGrid(models["GBT"].maxDepth, [3, 5])
            .build(),
    }

    for name in models:
        print(f"Training {name}...")
        cv = CrossValidator(
            estimator=models[name],
            estimatorParamMaps=param_grids[name],
            evaluator=evaluator,
            numFolds=5,
            seed=42
        )

        cv_model = cv.fit(train)
        best_model = cv_model.bestModel
        auc = evaluator.evaluate(best_model.transform(test))
        print(f"{name} Best AUC: {auc:.4f}")
        print(f"{name} Params: {best_model.extractParamMap()}")

# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
