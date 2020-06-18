import pathlib

import numpy as np
import click
from dtoolbioimage import scale_to_uint8, Image as dbiImage

from aiutils.data import ImageMaskDataSet


def visualise_masks(im, mask, pred_mask, strings):

    im_numpy = np.transpose(im.numpy(), (1, 2, 0))

    mask_uint8 = scale_to_uint8(mask.numpy())
    pred_mask_uint8 = scale_to_uint8(pred_mask)

    rdim, cdim = pred_mask.shape
    mask_vis = np.zeros((rdim, cdim, 3), dtype=np.uint8)

    mask_vis[:,:,0] = pred_mask_uint8
    mask_vis[:,:,1] = mask_uint8

    merged = 0.5 * mask_vis + 0.5 * scale_to_uint8(im_numpy)

    pilim = annotate_with_strings(merged, strings)

    return pilim


def visualise_image_and_mask(im, mask):

    im_numpy = scale_to_uint8(np.transpose(im.numpy(), (1, 2, 0)))
    mask_rgb = np.dstack(3 * [scale_to_uint8(mask.numpy().squeeze())])

    merged = 0.5 * im_numpy + 0.5 * mask_rgb

    return merged.view(dbiImage)
  

@click.command()
@click.argument('imds_uri')
@click.argument('output_dirpath')
def main(imds_uri, output_dirpath):

    imds = ImageMaskDataSet(imds_uri)
    output_dirpath = pathlib.Path(output_dirpath)

    im, mask = imds[0]

    vis = visualise_image_and_mask(im, mask)

    vis.save(output_dirpath / "vis.png")



if __name__ == "__main__":
    main()