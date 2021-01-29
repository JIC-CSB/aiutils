import numpy as np
import skimage.draw

from dtoolbioimage import scale_to_uint8, Image as dbiImage


def visualise_image_and_mask(im, mask, mask_weight=0.5):

    im_numpy = scale_to_uint8(im)
    mask_rgb = np.dstack(3 * [scale_to_uint8(mask)])

    im_weight = 1 - mask_weight
    merged = im_weight * im_numpy + mask_weight * mask_rgb

    return merged.view(dbiImage)


def annotate_image_with_points(im, points, colour=[255, 0, 0]):

    ann = im.copy()

    for p in points:
        rr, cc = skimage.draw.disk(p, 5)
        ann[rr, cc] = colour

    return ann

