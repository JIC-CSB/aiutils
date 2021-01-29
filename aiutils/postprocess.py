import io

import numpy as np
import imageio
import skimage.filters
import skimage.measure

from dtoolbioimage import scale_to_uint8





class BinaryMask(np.ndarray):

    # def centroid(self, rid=0):
    #     label_im = skimage.measure.label(self)

    def _repr_png_(self):
        b = io.BytesIO()
        scaled = scale_to_uint8(self)
        imageio.imsave(b, scaled, 'PNG', compress_level=0)

        return b.getvalue()

    @property
    def borders(self):
        eroded_mask = skimage.morphology.erosion(self)
        border_mask = eroded_mask ^ self
        return border_mask.view(BinaryMask)


class RegionMask(BinaryMask):

    @classmethod
    def from_array(cls, array):
        region = array.view(cls)
        label_im = skimage.measure.label(region)
        region.rprops = skimage.measure.regionprops(label_im)[0]

        return region

    @property
    def centroid(self):
        rf, cf = self.rprops.centroid

        return np.array((int(rf), int(cf)))


def binary_mask_from_model(unet_model, im):
    pred_mask = unet_model.scaled_mask_from_image(im)
    return binarise_mask(pred_mask)


def largest_mask_region_from_model(unet_model, im):
    mask = binary_mask_from_model(unet_model, im)

    return largest_mask_region(mask)


def binarise_mask(mask):
    thresh = skimage.filters.threshold_otsu(mask)
    binary_mask = mask > thresh

    return binary_mask.view(BinaryMask)


def largest_mask_region(mask):

    mask_label = skimage.measure.label(mask)
    rprops = skimage.measure.regionprops(mask_label)

    by_area = {r.area: r.label for r in rprops}
    largest_area = sorted(by_area, reverse=True)[0]

    return RegionMask.from_array(mask_label == by_area[largest_area])


def binary_mask_borders(binary_mask):
    eroded_mask = skimage.morphology.erosion(binary_mask)
    border_mask = eroded_mask ^ binary_mask
    return border_mask
