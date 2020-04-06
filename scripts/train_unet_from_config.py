import os
import logging
import pathlib

import click
import torch
import dtoolcore
import ruamel.yaml

from dtoolbioimage import Image as dbiImage

from dtoolai.parameters import Parameters
from dtoolai.training import train_model_with_metadata_capture
from dtoolai.utils import BaseCallBack

from aiutils.data import (
    LimitDataSetWrapper,
    ImageMaskDataSet,
    AIModelDataSetCreator
)
from aiutils.unetmodel import UNet, dice_loss
from aiutils.utils import composite_image_from_model_and_dataset
from aiutils.config import YAMLConfig


class ProcessDSCallback(BaseCallBack):

    def __init__(self, ds_uri, output_dirpath):
        self.ds = dtoolcore.DataSet.from_uri(ds_uri)

        self.output_dirpath = pathlib.Path(output_dirpath)
        self.output_dirpath.mkdir(exist_ok=True, parents=True)

    def on_epoch_end(self, epoch, model, history):
        output_fpath = self.output_dirpath / f"epoch{epoch}.png"
        logging.info(f"Writing composite image to {output_fpath}")
        procim = composite_image_from_model_and_dataset(model, self.ds)
        procim.view(dbiImage).save(output_fpath)


@click.command()
@click.argument('config_fpath')
def main(config_fpath):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TrainUNet")

    model_params = Parameters(
        batch_size=4,
        learning_rate=0.01,
        n_epochs=1
    )

    logger.info(f"Loading config from {config_fpath}")
    config = YAMLConfig(config_fpath)
    model_params.parameter_dict.update(config.parameters)

    ids = ImageMaskDataSet(config.training_dataset_uri)
    if config.test_mode:
        train_ds = LimitDataSetWrapper(ids)
    else:
        train_ds = ids

    # TODO - these should be set in some more sensible way
    train_ds.dim = (512, 512)
    train_ds.input_channels = 3

    model = UNet(**config.model_init_params)
    loss_fn = dice_loss
    optim = torch.optim.Adam(model.parameters())

    callback = ProcessDSCallback(config.monitor_dataset_uri, "scratch/foo")
    with AIModelDataSetCreator(
        config.output_name,
        config.output_base_uri,
        train_ds
    ) as output_ds:
        train_model_with_metadata_capture(
            model, train_ds, optim, loss_fn, model_params, output_ds,
            callbacks=[callback]
        )




if __name__ == "__main__":
    main()
