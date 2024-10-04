import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow

def model_pipeline(data_test: pd.DataFrame, model_name) -> pd.DataFrame:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/2")
    
    scaler = StandardScaler()
    data_test[['price','qty','percent_to_1000']] = scaler.fit_transform(data_test[['price','qty','percent_to_1000']]).astype('float32')
    data_test[['isBuyerMaker', 'isBestMatch']]=data_test[['isBuyerMaker','isBestMatch']].astype('int32')
    
    predictions = model.predict(data_test)
    
    return predictions