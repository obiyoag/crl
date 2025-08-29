import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


class WBCAttDataset(Dataset):
    def __init__(self, data_dir, data_frame, concept_list, transform):
        super().__init__()
        self.data_dir = data_dir
        self.data_frame = data_frame
        self.concept_list = concept_list
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        sample = self.data_frame.iloc[index, :]
        file_path = "/".join(sample.path.split("/")[1:])
        image = Image.open(self.data_dir / "raw_data" / file_path)
        image = self.transform(image)
        concept = torch.FloatTensor(sample.loc[self.concept_list].tolist())
        return image, int(sample.label), concept


class WBCAttDataModule(pl.LightningDataModule):
    def __init__(self, cfgs):
        super().__init__()
        self.data_dir = Path(cfgs.data.data_dir)
        self.batch_size = cfgs.data.batch_size

        # load meta_data
        train_df = pd.read_csv(self.data_dir / "meta_data" / "train.csv")
        valid_df = pd.read_csv(self.data_dir / "meta_data" / "valid.csv")
        test_df = pd.read_csv(self.data_dir / "meta_data" / "test.csv")

        train_df["partition"] = "train"
        valid_df["partition"] = "valid"
        test_df["partition"] = "test"

        df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

        concepts = df.columns.tolist()[2:-2]
        df["label"] = df["label"].replace(cfgs.data.process_dicts["label"])

        new_dict = {}
        for concept in concepts:
            subtypes = df[concept].unique()
            for subtype in subtypes:
                key = f"{subtype} {concept}".replace("_", " ").replace("-", " ")
                value = (df[concept] == subtype).tolist()
                if len(subtypes) == 2 and key in cfgs.data.process_dicts["binarys"]:
                    key = key.replace("yes ", "")
                    new_dict[key] = value
                elif len(subtypes) > 2:
                    new_dict[key] = value

        concept_df = pd.DataFrame(new_dict, dtype=int)
        df = pd.concat([df.iloc[:, [-2, -1, 1]], concept_df], axis=1)
        df[df.columns[3:]] = df[df.columns[3:]].astype(float)
        self.concept_list = df.columns[3:].tolist()

        self.classes = [
            "Basophil",
            "Eosinophil",
            "Lymphocyte",
            "Monocyte",
            "Neutrophil",
        ]
        self.df = df

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
        if stage == "fit":
            train_df = self.df[self.df["partition"] == "train"]
            valid_df = self.df[self.df["partition"] == "valid"]
            self.train_dataset = WBCAttDataset(
                self.data_dir, train_df, self.concept_list, self.aug_transform
            )
            self.valid_dataset = WBCAttDataset(
                self.data_dir, valid_df, self.concept_list, self.noaug_transform
            )
        elif stage == "test":
            test_df = self.df[self.df["partition"] == "test"]
            self.test_dataset = WBCAttDataset(
                self.data_dir, test_df, self.concept_list, self.noaug_transform
            )
        elif stage == "predict":
            pass
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.batch_size, True, num_workers=4, pin_memory=True
        )

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset, self.batch_size, False, num_workers=4, pin_memory=True
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset, self.batch_size, False, num_workers=4, pin_memory=True
        )
        return test_loader

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=4, pin_memory=True)
