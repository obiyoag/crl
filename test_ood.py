import torch
import pandas as pd
import pytorch_lightning as pl

from PIL import Image
from pathlib import Path
from callbacks import ComputeMetric
from models.crl import CRL
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class DDIDataset(Dataset):
    def __init__(self, data_dir, data_frame):
        super().__init__()
        self.data_dir = data_dir
        self.data_frame = data_frame
        self.concept_list = data_frame.columns[2:50].tolist()
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                self.convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def convert_image_to_rgb(image):
        return image.convert("RGB")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        sample = self.data_frame.iloc[index, :]
        image = Image.open(self.data_dir / "raw_data" / f"{sample.id}")
        image = self.transform(image)
        concept = torch.FloatTensor(sample.loc[self.concept_list].tolist())
        return image, int(sample.label), concept


def main():
    data_dir = Path("/workspace/Datasets/DDI")

    # load data df
    data_df = pd.read_csv(data_dir / "meta_data" / "ddi_metadata.csv", index_col=0)
    data_df = data_df[["DDI_file", "malignant"]]
    data_df = data_df.rename(columns={"DDI_file": "id", "malignant": "label"})
    data_df["label"] = data_df["label"].map({True: 1, False: 0}).astype(int)
    # load concept df
    concept_df = pd.read_csv(data_dir / "meta_data/skincon.csv", index_col=0)
    # filter out images with poor quality
    concept_df = concept_df[concept_df["Do not consider this image"] != 1]
    concept_df = concept_df.drop("Do not consider this image", axis=1)
    concept_df = concept_df.rename(columns={"ImageID": "id"})
    # merge task label in the concept dataframe
    df = pd.merge(data_df[["id", "label"]], concept_df, on="id", how="inner")

    dataloder = DataLoader(DDIDataset(data_dir, df), 32, False, num_workers=4)

    model = CRL.load_from_checkpoint("checkpoints/skin_crl/last.ckpt")

    trainer = pl.Trainer(devices=[0], callbacks=ComputeMetric(), logger=False)
    trainer.test(model, dataloaders=dataloder)


if __name__ == "__main__":
    main()
