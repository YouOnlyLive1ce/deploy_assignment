import mlflow
import mlflow.xgboost
import xgboost as xgb
import pyarrow.parquet as pq
import pandas as pd
import os
import glob
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

def get_processed_data_paths():
    # Get all parquet files that match the pattern for training
    data_path = './data/processed/'
    files = glob.glob(os.path.join(data_path, 'train*_labeled.parquet'))
    return files

def read_and_concatenate_parquet_files():
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
    X_train = full_df.iloc[:, :-1]  # All columns except the last one
    y_train = full_df['class']      # Assuming 'class' is the label column

    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

# Enable automatic logging
mlflow.autolog()

# Read and prepare data
X_train, X_val, y_train, y_val = read_and_concatenate_parquet_files()
print(X_train.dtypes)
num_classes = len(y_train.value_counts())
print(f"Feature columns: {X_train.columns}")
print(f"Class distribution:\n{y_train.value_counts()}")

# Define the model and grid search
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 7],
    'n_estimators': [30, 50],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 1.0]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

# Start an MLflow experiment
experiment_name = f"xgb gridsearch {num_classes} classes"
run_name = "run 01"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
print(f"Experiment ID: {experiment_id}")

# Train and log model
with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    # Log metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)
    mlflow.set_tag("Training", "xgb")
    mlflow.set_tag("Dataset", "concatenated files")

    signature = infer_signature(X_train, y_train)
    model_info = mlflow.xgboost.log_model(
        xgb_model=best_model,
        artifact_path="xgb" + str(num_classes),
        signature=signature,
        input_example=X_train
    )

    print("Model saved with URI:", model_info.model_uri)
