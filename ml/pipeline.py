import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import os
import yaml
import subprocess

def model_pipeline(data_test: pd.DataFrame, model_name) -> pd.DataFrame:
    # Determine if running inside Docker by checking for a specific environment variable or file path
    if os.path.exists('/.dockerenv'):
        # Running inside Docker: Use the path relative to the Docker container
        meta_path = f"./mlruns/models/{model_name}/version-4/meta.yaml"
        subprocess.run(['ls', '-la', '.'])
        # Load the meta.yaml file to retrieve model source
        with open(meta_path, 'r') as meta_file:
            meta_data = yaml.safe_load(meta_file)
        
        # Retrieve the model source
        model_uri = meta_data['source']
        
        # Apply logic to extract the path after 'mlruns'
        idx = model_uri.find('mlruns')
        model_uri = model_uri[idx:]
    else:
        # Running locally: Use the path relative to the local project directory
        model_uri = f"models:/{model_name}/4"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    scaler = StandardScaler()
    data_test[['price','qty','percent_to_1000']] = scaler.fit_transform(data_test[['price','qty','percent_to_1000']]).astype('float32')
    
    predictions = model.predict(data_test)
    
    return predictions