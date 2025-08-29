from .skincon import SkinConDataModule
from .wbcatt import WBCAttDataModule


def get_datamodules(cfgs):
    if cfgs.data.dataset == "skincon":
        datamodule = SkinConDataModule(cfgs)
    elif cfgs.data.dataset == "wbcatt":
        datamodule = WBCAttDataModule(cfgs)
    else:
        raise NotImplementedError
    return datamodule
