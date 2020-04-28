import logging

import click
import torch

from dtoolai.parameters import Parameters
from dtoolai.training import train_model_with_metadata_capture

from aiutils.config import YAMLConfig
from aiutils.data import (
    AIModelDataSetCreator,
    ImageRegressionDataSet,
    LimitDataSetWrapper
)
from aiutils.regressormodel import RegressNet


def preflight(train_ds, model, loss_fn, optim):

    # print(model.__class__.__name__)

    logging.info(f"Model name {model.model_name}")
    logging.info(f"Training dataset has {len(train_ds)} items")


@click.command()
@click.argument('config_fpath')
def main(config_fpath):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TrainRegressor")

    model_params = Parameters(
        batch_size=4,
        learning_rate=0.01,
        n_epochs=1
    )

    logger.info(f"Loading config from {config_fpath}")
    config = YAMLConfig(config_fpath)
    model_params.parameter_dict.update(config.parameters)

    logger.info(f"Loading dataset from {config.training_dataset_uri}")
    irds = ImageRegressionDataSet(config.training_dataset_uri)

    if config.test_mode:
        train_ds = LimitDataSetWrapper(irds)
    else:
        train_ds = irds

    # TODO - these should be set in some more sensible way
    train_ds.dim = (512, 512)
    train_ds.input_channels = 3

    model = RegressNet(input_channels=3)
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    preflight(train_ds, model, loss_fn, optim)

    with AIModelDataSetCreator(
        config.output_name,
        config.output_base_uri,
        irds
    ) as output_ds:
        train_model_with_metadata_capture(
            model, train_ds, optim, loss_fn, model_params, output_ds
        )


if __name__ == "__main__":
    main()
