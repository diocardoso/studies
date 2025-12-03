import os
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from deep_learning.handwritten_digit_classification.nn import NNet
from deep_learning.handwritten_digit_classification.preprocess import (
    load_mnist_images,
    load_mnist_labels,
    convert_images_to_4d_tensor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load MNIST dataset path from environment variable
MNIST_DATASET_PATH = Path(os.environ.get("MNIST_DATASET_PATH"))


def main():
    logger.info("Loading MNIST dataset...")
    test_images = load_mnist_images(MNIST_DATASET_PATH / "t10k-images.idx3-ubyte")
    test_labels = load_mnist_labels(MNIST_DATASET_PATH / "t10k-labels.idx1-ubyte")

    assert len(test_images) == len(test_labels), "Mismatch in test data"
    logger.info(f"Test samples: {len(test_images)}")

    # Convert images and labels to tensors
    test_images_tensor = convert_images_to_4d_tensor(test_images)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    # Initialize network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NNet().to(device)

    # Save trained model
    trained_model = "mnist.pth"
    net.load_state_dict(torch.load(trained_model, weights_only=True))

    # Wrap data in DataLoader for batch iteration
    test_loader = DataLoader(
        TensorDataset(test_images_tensor, test_labels_tensor),
        batch_size=128,
        shuffle=True,
    )
    correct = 0
    total = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )


if __name__ == "__main__":
    main()
