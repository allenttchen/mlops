import os
import json
import argparse

import mlflow
import numpy as np
from src.utils import set_device
import torch
from torch import optim, nn
from src.utils import read_params
from src.dataloaders import load_mnist_dataloader
from src.models import CNN
from src.trainers import Trainer


def train_model(model_name, config_path):
    """
    Run a round of model training with experiment loggings
    """
    # Configs
    config = read_params(config_path)
    training_config = config["training"]
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # load data
    device = set_device(mps=True)
    train_dataloader, val_dataloader = load_mnist_dataloader(
        root="data", flatten=False, batch_size=training_config["batch_size"]
    )

    # Training start
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as run:
        model = CNN()
        optimizer = optim.Adam(model.parameters(), lr=training_config["lr"])
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
        results = trainer.train(
            epochs=training_config["epochs"],
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        print(results)
        mlflow.log_params(training_config)
        mlflow.log_metrics({k+"_avg": np.mean(v) for (k, v) in results.items()})
        # log metric for each epoch
        for metric_name, metrics in results.items():
            for i in range(len(metrics)):
                mlflow.log_metric(metric_name+f"_{i}", metrics[i])

        # log on MLFlow
        mlflow.pytorch.log_state_dict(
            state_dict=trainer.model.state_dict(),
            artifact_path=mlflow_config["registered_model_path"]
        )

        # save model locally
        mlflow.pytorch.save_state_dict(
            state_dict=trainer.model.state_dict(),
            path=mlflow_config["registered_model_path"],
        )


        # # Save Metrics
        # logging_config = config["logging"]
        # os.makedirs(logging_config["metrics_dir"], exist_ok=True)
        # metrics_path = os.path.join(logging_config["metrics_dir"], logging_config["metrics_path"])
        # with open(metrics_path, "w") as f:
        #     json.dump(results, f, indent=4)
        #
        # # Save model
        # os.makedirs(logging_config["model_dir"], exist_ok=True)
        # model_path = os.path.join(logging_config["model_dir"], logging_config["model_path"])
        # torch.save(trainer.model.state_dict(), model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_model("CNN", config_path=parsed_args.config)
