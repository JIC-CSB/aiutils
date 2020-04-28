import itertools

import torch
import numpy as np

from skimage.transform import resize
from dtoolbioimage import Image as dbiImage
from torchvision.transforms.functional import to_tensor


def fpath_to_composite_image(model, fpath, dim=(512, 512)):
    im = dbiImage.from_file(fpath)
    scaled_im = resize(im, dim).astype(np.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = to_tensor(scaled_im)
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        pred_mask_tensor = model(input_tensor[None])

        scaled_mask = pred_mask_tensor.squeeze().cpu().numpy()

    mask = resize(scaled_mask, im.shape)

    im[np.nonzero(mask[:, :, 0] > 0.5)] = 255, 0, 0

    return im


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def composite_image_from_model_and_dataset(model, ds):
    fpath_iter = (ds.item_content_abspath(idn) for idn in ds.identifiers)

    def make_thumbnail(im): return resize(im, (256, 256))
    def fpath_to_im(fpath): return fpath_to_composite_image(model, fpath)
    composite_images = map(fpath_to_im, fpath_iter)
    thumbnails = map(make_thumbnail, composite_images)
    # TODO - messy hard coding
    grouped = grouper(thumbnails, 6)
    tiled = np.vstack([np.hstack(group) for group in grouped])

    return tiled.view(dbiImage)
