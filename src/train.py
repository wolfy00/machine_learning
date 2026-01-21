# src/train.py
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from common import read_feature_table, set_mlflow_for_uc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="UC catalog (dev_ml/staging_ml/prod_ml)")
    parser.add_argument("--schema", default="features")
    parser.add_argument("--table", required=True, help="feature table name")
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--model-name", required=True, help="UC model name: catalog.schema.model_name")
    parser.add_argument("--n-estimators", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    set_mlflow_for_uc()

    # 1. Load data from Unity Catalog
    df_spark = read_feature_table(args.catalog, args.schema, args.table)
    df = df_spark.toPandas()

    X = df.drop(columns=[args.label_col])
    y = df[args.label_col]  # <-- small bug; fix to args.label_col
