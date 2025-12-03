from pathlib import Path

import numpy as np
import struct
from array import array

import torch


def load_mnist_images(images_fp: Path) -> list[np.ndarray]:
    with open(images_fp, "rb") as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        assert magic == 2051, f"Magic number mismatch, expected 2051, got {magic}"
        image_data = array("B", file.read())

    images = [
        np.array(image_data[i * rows * cols : (i + 1) * rows * cols]).reshape(28, 28)
        for i in range(size)
    ]
    return images


def load_mnist_labels(labels_fp: Path) -> list[int]:
    with open(labels_fp, "rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        assert magic == 2049, f"Magic number mismatch, expected 2049, got {magic}"
        labels = array("B", file.read()).tolist()

    return labels


def convert_images_to_4d_tensor(images: list[np.ndarray]) -> torch.Tensor:
    images = np.stack(images)
    tensor = torch.tensor(images, dtype=torch.float32)

    # add channel dimension
    tensor = tensor.unsqueeze(1)

    # pad images to 32x32
    tensor = torch.nn.functional.pad(tensor, pad=(2, 2, 2, 2), mode="constant", value=0)

    # normalize
    tensor = tensor / 255.0
    return tensor
