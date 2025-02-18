from typing import List, Tuple
import numpy as np
import os
import glob
from PIL import Image
import math
import random


class DataLoader:
    """A base class for data loaders."""

    def __init__(
        self, path_to_data: str, batch_size: int = 512, split: str = "train"
    ) -> None:
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.inputs = []
        self.targets = []
        self.size = 0
        self.split = split
        self.batch_iter = 0
        self.n_batch = 0

    def download(self) -> None:
        """
        Downloads a dataset from the web to disk.

        """
        raise NotImplementedError(
            "load_into_memory() must be implemented in subclasses."
        )

    def load_into_memory(self) -> None:
        """Loads the entire dataset into memory"""
        raise NotImplementedError(
            "load_into_memory() must be implemented in subclasses."
        )

    def get_next_batch(self) -> Tuple[List, List]:
        """
        Grabs the next batch of data.

        Returns:
            A Tuple of the list of inputs and list of target labels
            inside the next batch.
        """
        start = self.batch_size * self.batch_iter
        end = self.batch_size * (self.batch_iter + 1)
        end = min(end, self.size)
        to_take = np.arange(start, end)
        self.batch_iter = (self.batch_iter + 1) % self.n_batch
        return (
            [self.inputs[idx] for idx in to_take],
            [self.targets[idx] for idx in to_take],
        )

    def shuffle(self) -> None:
        """
        Shuffles the order of the dataset randomly.
        """
        data = list(zip(self.inputs, self.targets))
        random.shuffle(data)
        self.inputs, self.targets = zip(*data)


class MNISTLoader(DataLoader):
    """A dataloader for the MNIST dataset of 60K images of handwritten digits."""

    def __init__(
        self, path_to_data: str, batch_size: int = 512, split: str = "train"
    ) -> None:
        super().__init__(path_to_data, batch_size, split)

    def download(self) -> None:
        """
        Downloads the dataset from the web to disk.
        """
        print(f"Cloning MNIST into {self.path_to_data}")
        os.system(
            f"git clone https://github.com/rasbt/mnist-pngs.git {self.path_to_data}"
        )

    def load_into_memory(self) -> None:
        """
        Copies the dataset from disk into memory.
        """
        # check if the repo exists at the path
        if not os.path.exists(self.path_to_data):
            self.download()

        data_dir = f"{self.path_to_data}/{self.split}"
        label_set = [
            os.path.basename(d) for d in glob.glob(f"{data_dir}/*") if os.path.isdir(d)
        ]
        # sort the label set and enumerate
        label_set.sort()

        label_map = {}
        for i, l in enumerate(label_set):
            label_map[l] = i

        for label in label_set:
            img_paths = [f for f in glob.glob(f"{data_dir}/{label}/*.png")]

            for img_path in img_paths:
                img = Image.open(img_path)
                self.inputs.append(img)
                self.targets.append(label_map[label])

        self.size = len(self.targets)
        self.n_batch = math.ceil(self.size / self.batch_size)
