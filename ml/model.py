import pandas as pd
import mlflow

def load_model(run_id,model_artifact_path):
    """Load a pre-trained model.

    Returns:
        model (function): A function that takes a input and returns class.
    """
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/{model_artifact_path}")

    def model_fn(data_test: pd.DataFrame) -> pd.DataFrame:
        pred = loaded_model.predict(data_test)
        return pred

    return model_fn