import os
import json
import argparse

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
    config = read_params(config_path)
    device = set_device(mps=True)
    train_dataloader, val_dataloader = load_mnist_dataloader(
        root="data", flatten=False, batch_size=config["training"]["batch_size"]
    )
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
    results = trainer.train(
        epochs=config["training"]["epochs"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    print(results)

    # TODO: Replace the below with MLFlow
    # Save Metrics
    logging_config = config["logging"]
    os.makedirs(logging_config["metrics_dir"], exist_ok=True)
    metrics_path = os.path.join(logging_config["metrics_dir"], logging_config["metrics_path"])
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save model
    os.makedirs(logging_config["model_dir"], exist_ok=True)
    model_path = os.path.join(logging_config["model_dir"], logging_config["model_path"])
    torch.save(trainer.model.state_dict(), model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_model("CNN", config_path=parsed_args.config)
