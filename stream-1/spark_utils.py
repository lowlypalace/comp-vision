from ast import literal_eval
import os
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
import matplotlib.patches as mpatches

from torch.utils.data import Dataset
from torch import tensor, from_numpy

class SPARKDataset:

    """Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data."""

    def __init__(self, class_map, root_dir="./data", split="train"):
        self.root_dir = os.path.join(root_dir, split)
        self.labels = self.process_labels(root_dir, split)
        self.class_map = class_map

    def __len__(self):
        return len(self.labels)

    def process_labels(self, labels_dir, split):
        path = os.path.join(labels_dir, f"{split}.csv")
        labels = pd.read_csv(path)
        return  labels

    def get_image(self, i=0):
        """Loading image as PIL image."""

        sat_name = self.labels.iloc[i]["class"]
        img_name = self.labels.iloc[i]["filename"]
        image_name = f"{self.root_dir}/{img_name}"

        image = io.imread(image_name)

        return image, self.class_map[sat_name]

    def get_bbox(self, i=0):
        """Getting bounding box for image."""

        bbox = self.labels.iloc[i]["bbox"]
        bbox = literal_eval(bbox)

        min_x, min_y, max_x, max_y = bbox

        return min_x, min_y, max_x, max_y

    def visualize(self, i, size=(15, 15), ax=None):
        """Visualizing image, with ground truth pose with axes projected to training image."""

        if ax is None:
            ax = plt.gca()

        image, img_class = self.get_image(i)
        min_x, min_y, max_x, max_y = self.get_bbox(i)

        ax.imshow(image, vmin=0, vmax=255)

        rect = mpatches.Rectangle(
            (min_y, min_x),
            max_y - min_y,
            max_x - min_x,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

        label = f"{list(self.class_map.keys())[list(self.class_map.values()).index(img_class)]}"

        ax.text(min_y, min_x - 20, label, color="white", fontsize=15)
        ax.set_axis_off()

        return

class PyTorchSparkDataset(Dataset):

    """SPARK dataset that can be used with DataLoader for PyTorch training."""
    def __init__(
        self,
        class_map,
        split="train",
        root_dir="",
        transform=None,
        sample_size=1,
    ):

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "Invalid split, has to be either 'train', 'validation' or 'test'"
            )
        if split == "test" and sample_size != 1:
            raise ValueError("Cannot sample from test set")

        self.class_map = class_map
        self.split = split
        self.root_dir = os.path.join(root_dir, self.split)
        self.labels =self.process_labels(root_dir, split, sample_size)
        self.transform = transform

    def process_labels(self, labels_dir, split, sample_size):
        path = os.path.join(labels_dir, f"{split}.csv")
        labels = pd.read_csv(path)

        # Check if sample_size is in the correct range
        if not 0 <= sample_size <= 1:
            raise ValueError("sample_size must be between 0 and 1")
        elif sample_size == 1:
            return labels
        else:
            # Group the dataframe by class and sample from each group
            sampled_labels = (
                labels.groupby("class")
                .apply(lambda x: x.sample(frac=sample_size, random_state=1))
                .reset_index(drop=True)
            )
            return sampled_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx]["filename"]
        image_name = f"{self.root_dir}/{img_name}"

        image = io.imread(image_name)
        torch_image = from_numpy(image).permute(2, 1, 0)

        if self.split == 'test':
            return torch_image, img_name

        else:
            sat_name = self.labels.iloc[idx]["class"]
            bbox = self.labels.iloc[idx]["bbox"]
            bbox = literal_eval(bbox)
            return torch_image, self.class_map[sat_name], tensor(bbox)
