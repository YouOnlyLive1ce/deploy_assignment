from pathlib import Path
import yaml
import pickle
import pandas as pd

# load config file
config_path = Path(__file__).parent.parent / "configs/config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

def load_model(model_name):
    """Load a pre-trained model.

    Returns:
        model (function): A function that takes a input and returns class.
    """
    model_path= Path(__file__).parent.parent / "models" / f"{model_name}.pkl"
    model = pickle.load(open(model_path, 'rb'))

    def model_fn(data_test: pd.DataFrame) -> pd.DataFrame:
        pred = model.predict(data_test)
        return pred

    return model_fn