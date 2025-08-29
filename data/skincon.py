import json
import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


class SkinDataset(Dataset):
    def __init__(self, data_dir, data_dict, transform):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.images, self.concepts, self.labels = [], [], []
        for item in data_dict:
            self.images.append(item["id"])
            self.labels.append(item["label"])
            self.concepts.append(torch.tensor(item["concepts"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.data_dir / "raw_data" / self.images[index])
        image = self.transform(image)
        label = self.labels[index]
        concepts = self.concepts[index]
        return image, label, concepts


class SkinConDataModule(pl.LightningDataModule):
    def __init__(self, cfgs):
        super().__init__()
        self.data_dir = Path(cfgs.data.data_dir)
        self.batch_size = cfgs.data.batch_size

        df = pd.read_csv(self.data_dir / "meta_data" / "skincon.csv")
        self.concept_list = df.columns[2:50].tolist()
        self.classes = ["benign", "malignant"]

        # load meta_data
        with open(self.data_dir / "meta_data" / "5_fold_splits.json", "r") as f:
            self.data_dict = json.load(f)[f"fold_{cfgs.data.fold}"]

        self.aug_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(size=cfgs.data.img_size),
                transforms.ToTensor(),
                transforms.Normalize(cfgs.data.img_mean, cfgs.data.img_std),
            ]
        )
        self.noaug_transform = transforms.Compose(
            [
                transforms.Resize(size=cfgs.data.img_size),
                transforms.CenterCrop(size=cfgs.data.img_size),
                transforms.ToTensor(),
                transforms.Normalize(cfgs.data.img_mean, cfgs.data.img_std),
            ]
        )

    def setup(self, stage):
        self.train_dataset = SkinDataset(
            self.data_dir, self.data_dict["train"], self.aug_transform
        )
        self.valid_dataset = SkinDataset(
            self.data_dir, self.data_dict["valid"], self.noaug_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.batch_size, True, num_workers=4, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, self.batch_size, num_workers=4, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.valid_dataset, self.batch_size, num_workers=4, pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=4, pin_memory=True)
