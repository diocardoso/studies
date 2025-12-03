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
    train_images = load_mnist_images(MNIST_DATASET_PATH / "train-images.idx3-ubyte")
    train_labels = load_mnist_labels(MNIST_DATASET_PATH / "train-labels.idx1-ubyte")

    assert len(train_images) == len(train_labels), "Mismatch in training data"
    logger.info(f"Training samples: {len(train_images)}")

    # Convert images and labels to tensors
    train_images_tensor = convert_images_to_4d_tensor(train_images)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

    # Wrap data in DataLoader for batch iteration
    train_loader = DataLoader(
        TensorDataset(train_images_tensor, train_labels_tensor),
        batch_size=32,
        shuffle=True,
    )

    # Initialize network, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        for batch_images, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch_images, batch_labels = (
                batch_images.to(device),
                batch_labels.to(device),
            )

            optimizer.zero_grad()
            outputs = net(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - Average Loss: {epoch_loss / len(train_loader):.4f}"
        )

    # Save trained model
    trained_model = "mnist.pth"
    torch.save(net.state_dict(), trained_model)
    logger.info(f"Trained model saved: {trained_model}")


if __name__ == "__main__":
    main()
