
# SHARED DATA PIPELINE
# Used by Vivian, Joshua, and Devan.
# Do not change independently between experiments.

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

CLASS_TO_IDX = {
    class_name: idx
    for idx, class_name in enumerate(CLASS_NAMES)
}

IDX_TO_CLASS = {
    idx: class_name
    for class_name, idx in CLASS_TO_IDX.items()
}


class TrashNetDataset(Dataset):
    """
    Connects the fixed CSV split to the actual TrashNet image files.
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = row["filepath"]
        class_name = row["class"]

        image = Image.open(image_path).convert("RGB")
        label = CLASS_TO_IDX[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    split_csv_path,
    img_size=224,
    batch_size=32,
    training_seed=0
):
    """
    Creates the shared training, validation, and test DataLoaders.
    """

    split_df = pd.read_csv(split_csv_path)

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "validation"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = TrashNetDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = TrashNetDataset(
        val_df,
        transform=eval_transform
    )

    test_dataset = TrashNetDataset(
        test_df,
        transform=eval_transform
    )

    generator = torch.Generator()
    generator.manual_seed(training_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
