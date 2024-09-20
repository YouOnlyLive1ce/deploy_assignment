from pathlib import Path
import yaml
import pickle
import pandas as pd

# load config file
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

def load_model():
    """Load a pre-trained model.

    Returns:
        model (function): A function that takes a input and returns class.
    """
    model = pickle.load(open('../models/xgb700.pkl', 'rb'))

    def model(data_test: pd.DataFrame) -> pd.DataFrame:
        pred = model.predict(data_test)
        return pred

    return model