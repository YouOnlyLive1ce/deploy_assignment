# Test perfomace on new unseen data

import mlflow
import mlflow.xgboost
import pandas as pd
import os
import glob
import sys

def get_processed_data_path():
    data_path = './data/processed/test_BTCUSDT-aggTrades-2024-09-16_custom_features.parquet'
    return data_path

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python ml/test.py <model_hash> <model_name><day to test>")
        sys.exit(1)

    # Get model_name and model_version from command line arguments
    run_id = sys.argv[1]
    model_artifact_path = sys.argv[2]
    day_to_test = sys.argv[3]

    # Load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/{model_artifact_path}")
    print(loaded_model.metadata)

    # Load data
    test_files = get_processed_data_path()
    

    # Test on new data
    predictions = loaded_model.predict(X)
    eval_data = X.copy()
    eval_data['class'] = y
    eval_data['predictions'] = predictions

    results = mlflow.evaluate(
        data=eval_data,
        model_type="classifier",
        targets="label",  # True labels column
        predictions="class",  # Predictions column
        evaluators=["default"]
    )

    print(f"metrics:\n{results.metrics}")
    print(f"artifacts:\n{results.artifacts}")

if __name__ == "__main__":
    main()