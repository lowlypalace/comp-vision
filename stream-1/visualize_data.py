#!/usr/bin/env python
# coding: utf-8

# # SPARK Dataset
# 

# ## Imports

# In[ ]:


from spark_utils import PyTorchSparkDataset, SPARKDataset
from matplotlib import pyplot as plt
from random import randint

# In[24]:


import torch
import torchvision
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate

# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
torchvision.disable_beta_transforms_warning()

# ## Defining the Dataset
# 
# By default, the output structure of the dataset is not compatible with the models or the transforms (https://pytorch.org/vision/master/transforms.html#v1-or-v2-which-one-should-i-use). To overcome that, we wrap a `PyTorchSparkDataset` in`PyTorchSparkDatasetV2`.
# 
# In the code below, we are wrapping images and bounding boxes `torchvision.TVTensor classes` so that we will be able to apply torchvision built-in transformations for the given object detection and segmentation task. Namely, image tensors will be wrapped by `torchvision.tv_tensors.Image` and bounding boxes into `torchvision.tv_tensors.BoundingBoxes`. Our dataset now returns a target which is dict where the values are `TVTensors` (all are `torch.Tensor` subclasses).
# 
# We also make the dataset compliant with COCO requirements so that it will work for both training and evaluation codes from the COCO reference script.

# In[25]:


# Wrap a PyTorchSparkDataset dataset for usage with torchvision.transforms.v2
class PyTorchSparkDatasetV2(torch.utils.data.Dataset):
    def __init__(self, class_map, split="train", root_dir="./data/", transforms=None, sample_size=1):
        super().__init__()
        self.dataset = PyTorchSparkDataset(class_map, split=split, root_dir=root_dir, sample_size=sample_size)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, bbox, img_name = self.dataset[idx]

        img = tv_tensors.Image(img)

        bbox = tv_tensors.BoundingBoxes(
            bbox, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=F.get_size(img)
        )
        label = torch.tensor([label])
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        image_id = idx

        target = {
            "boxes": bbox,
            "labels": label,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, img_name

# ## Transforms
# 
# Let’s now define our pre-processing transforms. All the transforms know how to handle images, bouding boxes and masks when relevant.
# 
# Transforms are typically passed as the transforms parameter of the dataset so that they can leverage multi-processing from the `torch.utils.data.DataLoader`.
# 
# If the data is intended for training, a series of augmentation techniques are used. These include `RandomPhotometricDistort` to apply random color distortions (this helps the model generalize better across varying lighting conditions), and `RandomZoomOut` which randomly zooms out of the image (creating new perspectives and scales for the model to learn from).
# 
# The `RandomIoUCrop` performs a random crop based on the intersection-over-union (IoU) of the bounding boxes, and `RandomHorizontalFlip` and `RandomVerticalFlip` randomly flips the image horizontally and vertically (increasing the diversity of orientations). The `SanitizeBoundingBoxes` function adjusts and sanitizes the bounding boxes after these transformations are applied.
# 
# - http://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#transforms-v2-end-to-end-object-detection-segmentation-example
# - https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py

# In[26]:


# Define the transforms to be applied to the data.
def get_transform(split):
    transforms = []
    transforms.append(T.ToImage())

    if split == "train":
        transforms.append(T.RandomPhotometricDistort(p=0.5))
        transforms.append(
            T.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0})
        )
        transforms.append(T.RandomIoUCrop())
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.RandomVerticalFlip(p=0.5))
        transforms.append(T.Resize((64, 64)))  # Remove this line
        transforms.append(T.SanitizeBoundingBoxes())

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)

# ## Loading Datasets

# In[27]:


def get_dataset(split, class_map, data_path, sample_size):
    # We use the PyTorchSparkDatasetV2 class defined above.
    dataset = PyTorchSparkDatasetV2(
        class_map=class_map,
        split=split,
        root_dir=data_path,
        transforms=get_transform(split),
        sample_size=sample_size,
    )
    return dataset

# In[28]:


# Set up the path to a local copy of the SPARK dataset, labels csv files should be in the same directory.
# The image sets should be in /data/train, /data/validation and /data/test.
data_path = "./data/"

# Define the class map, this is a dictionary that maps the class names to integer labels.
class_map = {
    "proba_2": 0,
    "cheops": 1,
    "debris": 2,
    "double_star": 3,
    "earth_observation_sat_1": 4,
    "lisa_pathfinder": 5,
    "proba_3_csc": 6,
    "proba_3_ocs": 7,
    "smart_1": 8,
    "soho": 9,
    "xmm_newton": 10,
}

# Define the number of classes
num_classes = len(class_map)

# # Define the datasets for training validation and testing
dataset = get_dataset(
    split="train", class_map=class_map, data_path=data_path, sample_size=0.01
)
dataset_valid = get_dataset(
    split="validation", class_map=class_map, data_path=data_path, sample_size=0.01
)
dataset_test = get_dataset(
    split="test", class_map=class_map, data_path=data_path, sample_size=0.01
)

print(f"Number of training samples: {len(dataset)}")
print(f"Number of validation samples: {len(dataset_valid)}\n")
print(f"Number of test samples: {len(dataset_test)}\n")

# In[29]:


# Check dataset format for debugging purposes
sample = dataset[0]
image, target, img_name = sample

print(f"Image type: {type(image)}")
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print()
print(f"Target type: {type(target)}")
print("Target keys: ", list(target.keys()))
print()
print(f"Boxes type: {type(target['boxes'])}")
print(f"Boxes shape: {target['boxes'].shape}")
print()
print(f"Labels type: {type(target['labels'])}")
print(f"Labels shape: {target['labels'].shape}")
print(f"Labels dtype: {target['labels'].dtype}")

# ## Visualizing Images

# In[30]:


rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

# Note that we are using the SPARKDataset class here instead of the PyTorchSparkDatasetV2 class
ds = SPARKDataset(class_map, root_dir=data_path, split="train")

for i in range(rows):
    for j in range(cols):
        ds.visualize(randint(0, len(dataset)), size=(10, 10), ax=axes[i][j])
        axes[i][j].axis("off")

fig.tight_layout()

# ## Define Dataloaders

# In[31]:


# Define the batch size to be used.
batch_size = 2

# Define the dataloaders for training, validation and testing.
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

data_loader_valid = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

# ## Defining Model
# 
# We will be using Faster R-CNN V2. Faster R-CNN V2 is a model that predicts both bounding boxes and class scores for potential objects in the image. It works similarly to Faster R-CNN with ResNet-50 FPN backbone.
# 
# We will start from a model pre-trained on COCO and finetune it for our particular classes in order to perform transfer learning.
# 
# - https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
# - https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
# - https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# In[21]:


# Define the model
def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    # TODO: Experiment with other weights such as 'COCO_V1'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# ## Model Training
# 
# Below is the main function which performs the training and the validation.

# In[ ]:


# Train on the GPU or on the CPU, if a GPU is not available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# Move model to the right device
model.to(device)

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# And a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Let's train it for 5 epochs
num_epochs = 1

for epoch in range(num_epochs):
    # Train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    # Update the learning rate
    lr_scheduler.step()
    # Evaluate on the test dataset
    evaluate(model, data_loader_valid, device=device)

# ## Compute Predictions

# In[ ]:


# Switch the model to evaluation mode
model.eval()

print("Evaluation done") # Printed gets stucked here
print(len(data_loader_test))
print(data_loader_test)

# Open the csv file
with open("predictions.csv", "w") as f:
    f.write("filename,class,bbox\n")
    # Loop over the test dataset
    for i, (images, targets, img_name) in enumerate(data_loader_test):
        # Move the images to the device
        images = list(image.to(device) for image in images) ### Always 2 images and 2 predictions why ??
        # Compute the model predictions
        with torch.no_grad():
            predictions = model(images)

        img_name_index = 0
        print("Len predictions", len(predictions))
        print("Len img_name", len(img_name), img_name)

        # Loop over the predictions
        for prediction in predictions:
            # Get the predicted boxes, labels and scores
            boxes = prediction["boxes"].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()
            scores = prediction["scores"].cpu().numpy()
            # Write the predictions to the csv file
            for box, label in zip(boxes, labels):
                # Convert the bounding box coordinates to integers
                box = list(map(int, box))
                # f.write(f"{dataset_valid.imgs[i]},{label},{box}\n")
                f.write(f"{img_name[img_name_index]},{label},{box}\n")
            img_name_index += 1

# In[ ]:


# # cd compvision_env/comp-vision/stream-1/

# torchrun --nproc_per_node=1 train.py\


# python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
#   train.py --model fasterrcnn_resnet50_fpn_v2 \
#   --train-sample-size 0.05 --test-sample-size 0.05 \
#   --epochs 26 --batch-size 2 --lr 0.02 --world-size $NGPU

