import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Spark (already available in Databricks)
spark = SparkSession.builder.getOrCreate()

# Tables
INPUT_TABLE = "default.input_features"
PREDICTION_TABLE = "default.model_predictions"
EXPERIMENT_NAME = "/Shared/github-mlflow-demo"
MODEL_NAME = "github_demo_model"

# -----------------------------
# 1. Read data from Databricks
# -----------------------------
df = spark.read.table(INPUT_TABLE)

pdf = df.toPandas()
X = pdf[["f1", "f2"]]
y = pdf["label"]

# -----------------------------
# 2. Train & register model
# -----------------------------
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="train-and-register"):
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)

    acc = accuracy_score(y, model.predict(X))

    mlflow.log_param("n_estimators", 20)
    mlflow.log_metric("accuracy", acc)

    # Register model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

# -----------------------------
# 3. Load model from registry
# -----------------------------
model_uri = f"models:/{MODEL_NAME}/latest"
loaded_model = mlflow.sklearn.load_model(model_uri)

# -----------------------------
# 4. Predict using registered model
# -----------------------------
pdf["prediction"] = loaded_model.predict(X)

# -----------------------------
# 5. Write predictions back to Databricks
# -----------------------------
pred_df = spark.createDataFrame(pdf)

(
    pred_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(PREDICTION_TABLE)
)

print("Training, registration, prediction completed successfully")
