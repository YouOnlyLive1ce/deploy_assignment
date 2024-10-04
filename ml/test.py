import mlflow
import mlflow.pyfunc
import pyarrow.parquet as pq
import pandas as pd
import os
import glob
import sys
from mlflow.tracking import MlflowClient

def get_processed_data_paths():
    # Get all parquet files that match the pattern for testing
    data_path = './data/processed/'
    files = glob.glob(os.path.join(data_path, 'test*_labeled.parquet'))
    return files

def load_test_data():
    files = get_processed_data_paths()
    print(f"Found files: {files}")

    # List to hold all dataframes
    dfs = []
    
    for file in files:
        table = pq.ParquetDataset(file).read()
        df = table.to_pandas()
        dfs.append(df)
    
    # Concatenate all the dataframes
    full_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Split features and labels
    X_test = full_df.iloc[:, :-1]  # All columns except the last one
    y_test = full_df['class']      # Assuming 'class' is the label column
    
    return X_test, y_test

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python test.py <model_hash> <model_name>")
        sys.exit(1)

    # Get model details from command line arguments
    run_id = sys.argv[1]
    model_artifact_path = sys.argv[2]

    # Load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/{model_artifact_path}")
    print(f"Model metadata: {loaded_model.metadata}")

    # Load test data
    X_test, y_test = load_test_data()
    print(f"Class distribution in the test set:\n{y_test.value_counts()}")

    # Make predictions
    predictions = loaded_model.predict(X_test)
    
    # Evaluate
    eval_data = X_test.copy()
    eval_data['label'] = y_test
    eval_data['predictions'] = predictions

    results = mlflow.evaluate(
        data=eval_data,
        model_type="classifier",
        targets="label",
        predictions="predictions",
        evaluators=["default"]
    )
    
    # Print metrics and artifacts
    print(f"Metrics:\n{results.metrics}")
    print(f"Artifacts:\n{results.artifacts}")

    # Register model if it passes the evaluation criteria
    if results.metrics['accuracy_score'] > 0.95 and results.metrics['f1_score'] > 0.95:
        mlflow.register_model(f"runs:/{run_id}/model", "xgb")
        client = MlflowClient()
        client.transition_model_version_stage(
            name="xgb",
            version=2,
            stage="Production"
        )
        print("New model registered as production version")

if __name__ == "__main__":
    main()
