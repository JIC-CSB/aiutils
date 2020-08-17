import os

import click
import dtoolcore

from dtoolbioimage import Image as dbiImage

from aiutils.vis import visualise_image_and_mask
from aiutils.unetmodel import TrainedUNet



def filtered_by_relpath(ds, relpath_filter):
    for idn in ds.identifiers:
        relpath = ds.item_properties(idn)['relpath']
        if relpath_filter(relpath):
            yield idn


def is_image(relpath):

    image_exts = [".png", ".jpg", ".tif", ".tiff"]
    _, ext = os.path.splitext(relpath)
    return ext in image_exts


class ImageDataSet(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.image_idns = list(filtered_by_relpath(dataset, is_image))

    @classmethod
    def from_uri(cls, uri):
        dataset = dtoolcore.DataSet.from_uri(uri)
        return cls(dataset)

    def __len__(self):
        return len(self.image_idns)

    def __getitem__(self, idx):
        idn = self.image_idns[idx]
        fpath = self.dataset.item_content_abspath(idn)
        return dbiImage.from_file(fpath)


def unetmodel_and_im_to_mask_vis(unetmodel, im):
    pred_mask = unetmodel.predict_mask_from_image(im)
    return visualise_image_and_mask(im, pred_mask)


@click.command()
@click.argument('model_uri')
@click.argument('dataset_uri')
def main(model_uri, dataset_uri):

    # TODO - shouldn't need input_channels?
    unetmodel = TrainedUNet(model_uri, input_channels=3)

    ids = ImageDataSet.from_uri(dataset_uri)

    merged_images = (
        unetmodel_and_im_to_mask_vis(unetmodel, im) 
        for im in ids
    )

    import pathlib
    dirpath = pathlib.Path('scratch')
    for n, mim in enumerate(merged_images):
        mim.save(dirpath/f"mim{n:02d}.png")


if __name__ == "__main__":
    main()