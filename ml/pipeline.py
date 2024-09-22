import pandas as pd
from sklearn.preprocessing import StandardScaler
from .model import load_model

def model_pipeline(data_test: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model_fn = load_model(model_name)
    
    # Preprocess the data (example: standard scaling)
    scaler = StandardScaler()
    data_test[['price','qty']] = scaler.fit_transform(data_test[['price','qty']]).astype('float32')
    
    # Make predictions
    predictions = model_fn(data_test)
    
    return predictions

# TODO: transformer scaler to fix curve ditribution