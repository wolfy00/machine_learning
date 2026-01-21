# src/common.py
import mlflow
from pyspark.sql import SparkSession

def get_spark():
    return SparkSession.builder.getOrCreate()

def read_feature_table(catalog: str, schema: str, table: str):
    spark = get_spark()
    full_name = f"{catalog}.{schema}.{table}"
    return spark.table(full_name)

def set_mlflow_for_uc():
    # Make sure we use Unity Catalog model registry
    mlflow.set_registry_uri("databricks-uc")
