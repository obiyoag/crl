import pytorch_lightning as pl

from models import CRL
from utils import setup_logger
from configs import get_configs
from data import get_datamodules
from callbacks import get_callbacks


def main(cfgs):
    logger = setup_logger(cfgs)
    model = CRL(cfgs)
    datamodule = get_datamodules(cfgs)
    callbacks = get_callbacks(cfgs)

    trainer = pl.Trainer(
        devices=[cfgs.device],
        logger=logger,
        max_epochs=cfgs.opt.max_epochs,
        log_every_n_steps=cfgs.log.log_every_n_steps,
        callbacks=callbacks,
        check_val_every_n_epoch=cfgs.log.val_every_n_epoch,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    # extract rules
    trainer.predict(model, datamodule=datamodule)


if __name__ == "__main__":
    cfgs = get_configs()
    main(cfgs)
