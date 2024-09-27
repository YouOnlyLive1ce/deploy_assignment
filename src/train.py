# big script for training. Includes data preprocessing calling, model saving

import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import os
import sys
import glob
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python src/train.py <day_to_train>")
        sys.exit(1)

    # Get model_name and model_version from command line arguments
    day_to_train = sys.argv[1]

    # Enable automatic logging
    mlflow.autolog()

    processed_folder = os.path.abspath(os.path.join('data', 'processed'))
    train_files = glob.glob(os.path.join(processed_folder, f'train*{day_to_train}.csv'))
    data_train = pd.concat([pd.read_csv(file) for file in train_files], ignore_index=True)

    X_train = data_train.iloc[:, :-1]
    y_train = data_train.iloc[:, -1]
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    num_classes=len(y_train.value_counts())
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
    experiment_name=f"xgb gridsearch {num_classes} classes"
    run_name="run 01"
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(experiment_id)

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        grid_search.fit(X_train_split, y_train_split)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)

        # Evaluate the model
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        print(f"Validation Accuracy: {accuracy}")

        # Log
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        mlflow.set_tag("Training","xgb","equal amount classes")

        signature = infer_signature(X_train, y_train)
        model_info = mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="xgb"+str(num_classes),
            signature=signature,
            input_example=X_train
        )

        print("Model saved with URI:", model_info.model_uri)

if __name__ == "__main__":
    main()