import os
import shutil

import click
import torch

from dtoolcore import DerivedDataSetCreator

from dtoolai.parameters import Parameters
from dtoolai.training import train_model_with_metadata_capture

from aiutils.unetmodel import UNet, dice_loss
from aiutils.data import ImageMaskDataSet


@click.command()
@click.argument('imds_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
@click.option('--params')
def main(imds_uri, output_base_uri, output_name, params):

    model_params = Parameters(
        batch_size=4,
        learning_rate=0.01,
        n_epochs=1
    )
    if params:
        model_params.update_from_comma_separated_string(params)

    train_ds = ImageMaskDataSet(imds_uri, usetype="training")
    valid_ds = ImageMaskDataSet(imds_uri, usetype="validation")

    # valid_len = int(0.2 * len(ds))
    # train_len = len(ds) - valid_len
    # train_ds, valid_ds = torch.utils.data.random_split(ds, [train_len, valid_len])
    # #FIXME ick
    # train_ds.uri = ds.uri
    # train_ds.name = ds.name
    # train_ds.uuid = ds.uuid

    model = UNet(input_channels=3)
    loss_fn = dice_loss
    optim = torch.optim.Adam(model.parameters())

    expected_dirpath = os.path.join(output_base_uri, output_name)
    if os.path.isdir(expected_dirpath):
        shutil.rmtree(expected_dirpath)
        
    with DerivedDataSetCreator(output_name, output_base_uri, train_ds) as output_ds:
        train_model_with_metadata_capture(model, train_ds, optim, loss_fn, model_params, output_ds, valid_ds)

if __name__ == "__main__":
    main()
