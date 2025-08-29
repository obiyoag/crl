import torch
import wandb
import torchvision
import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning.loggers import CSVLogger, WandbLogger


def setup_logger(cfgs):
    pl.seed_everything(cfgs.seed)

    Path(cfgs.log.save_dir).mkdir(parents=True, exist_ok=True)

    if cfgs.log.logger == "wandb":
        logger = WandbLogger(
            cfgs.exp_name,
            cfgs.log.save_dir,
            project=cfgs.project_name,
            config=cfgs,
            settings=wandb.Settings(code_dir="."),
        )
    elif cfgs.log.logger == "csv":
        logger = CSVLogger(cfgs.exp_name, cfgs.save_dir)
    else:
        logger = False
    return logger


def get_backbones(name, pretrained=True):
    weights = "DEFAULT" if pretrained else None
    model = getattr(torchvision.models, name)(weights=weights)
    if name.startswith("resnet"):
        n_features = list(model.modules())[-1].in_features
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported image encoder: {name}")
    return model, n_features
