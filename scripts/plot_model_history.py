"""Plot the history of a keras model from a dataset."""

import json

import click
import dtoolcore

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_idn_by_relpath(dataset, relpath):

    for idn in dataset.identifiers:
        if dataset.item_properties(idn)['relpath'] == relpath:
            return idn

    raise ValueError("Relpath {} not in dataset".format(relpath))


def item_content_abspath_from_relpath(dataset, relpath):

    idn = get_idn_by_relpath(dataset, relpath)

    return dataset.item_content_abspath(idn)


def plot_history_from_dataset(input_ds):

    history_fpath = item_content_abspath_from_relpath(input_ds, "history.json")


    with open(history_fpath) as fh:
        history = json.load(fh)

    # loss_array = np.array(history["loss"])
    # mean_loss_per_epoch = np.mean(loss_array, axis=1)
    # val_loss_array = np.array(history["val_loss"])
    # mean_val_loss_per_epoch = np.mean(val_loss_array, axis=1)


    # plt.plot(mean_loss_per_epoch)
    # plt.plot(mean_val_loss_per_epoch)

    plt.plot(history["dice_coeff"])
    plt.plot(history["val_dice_coeff"])
    plt.title("model performance")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    input_ds = dtoolcore.DataSet.from_uri(dataset_uri)

    plot_history_from_dataset(input_ds)


if __name__ == "__main__":
    main()
