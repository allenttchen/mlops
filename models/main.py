import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pytorch
from utils import set_device

mlflow_client = MlflowClient(

)


def run_model_training(model_name, hyperparameters, epochs):
    """
    Run a round of model training with experiment loggings
    """
    # Configs
    device = set_device(mps=True)
    mlflow.set_experiment("MNIST")

    #
