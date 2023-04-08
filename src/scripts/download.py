from pathlib import Path
import requests

from src.constants import (
    ROOT_DIR,
    MNIST_URL,
    MNIST_FILENAME
)


def main():

    local_mnist_path = Path(ROOT_DIR, "data/mnist")
    local_mnist_path.mkdir(parents=True, exist_ok=True)
    local_mnist_filepath = Path(local_mnist_path, MNIST_FILENAME)
    remote_mnist_filepath = MNIST_URL + MNIST_FILENAME

    if not local_mnist_filepath.exists():
        content = requests.get(remote_mnist_filepath).content
        with open(local_mnist_filepath, "wb") as file:
            file.write(content)


if __name__ == "__main__":
    main()
