from mlflow.pyfunc import load_model
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from src.utils import read_params

model_name = "MNISTModel"
stage = "Production"
version = "1"
#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflowclient = MlflowClient(
    "sqlite:///mlflow.db",
    "sqlite:///mlflow.db",
)
print(mlflow.get_tracking_uri())
print(mlflow.get_registry_uri())

model = load_model(model_uri=f"models:/{model_name}/{stage}")
data = np.random.rand(1, 28, 28).astype("float32")
pred = model.predict(data)[0]
res = int(np.argmax(pred))
print(res)

# all_model_versions = mlflowclient.search_model_versions(f"name='{model_name}'")
# last_model_version = all_model_versions[-1]
# curr_version = 1
# for model_version in all_model_versions:
#     if model_version.version > curr_version:
#         last_model_version = model_version
#
# print(last_model_version_for_none.name)
# print(last_model_version_for_none.version)



