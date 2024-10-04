from mlflow.tracking import MlflowClient
import mlflow
client = MlflowClient()

# Get a list of all registered models
registered_models = client.search_registered_models()

# Display the model names and details
for model in registered_models:
    print(f"Model name: {model.name}, Description: {model.description}")

model_name = "xgb"
model_version = 2

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

print("Model Metadata:", model.metadata)
print("Input Schema:", model.metadata.get_input_schema())
print("Output Schema:", model.metadata.get_output_schema())