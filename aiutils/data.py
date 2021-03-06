import os
import shutil

import torch
import dtoolcore
from torchvision.transforms.functional import to_tensor

from dtoolbioimage import Image
from dtoolai.data import WrappedDataSet


def add_image_mask_pair(output_ds, image, mask, n):
    image_relpath = 'images/image{:02d}.png'.format(n)
    mask_relpath = 'masks/mask{:02d}.png'.format(n)
    image_fpath = output_ds.prepare_staging_abspath_promise(image_relpath)
    mask_fpath = output_ds.prepare_staging_abspath_promise(mask_relpath)

    mask.view(Image).save(mask_fpath)
    image.view(Image).save(image_fpath)

    mask_id = dtoolcore.utils.generate_identifier(mask_relpath)

    metadata_appends = [
        (image_relpath, "mask_idn", mask_id),
        (mask_relpath, "mask_idn", None),
        (image_relpath, "is_image", True),
        (mask_relpath, "is_image", False)
    ]

    return metadata_appends


class ImageRegressionDataSet(WrappedDataSet):

    def __init__(self, ds_uri):
        super().__init__(ds_uri)

        self.coords_overlay = self.dataset.get_overlay('coords')
        self.im_identifiers = list(self.dataset.identifiers)

    def __getitem__(self, idx):
        idn = self.im_identifiers[idx]
        image_fpath = self.dataset.item_content_abspath(idn)
        im = Image.from_file(image_fpath)

        (r0, c0), (r1, c1) = self.coords_overlay[idn]
        tensor_coords = torch.Tensor([r0, c0, r1, c1])

        return to_tensor(im), tensor_coords

    def __len__(self):
        return len(self.im_identifiers)


class ImageMaskDataSet(WrappedDataSet):
    
    def __init__(self, ds_uri, usetype=None):
        super().__init__(ds_uri)

        is_image_overlay = self.dataset.get_overlay('is_image')
        mask_ids_overlay = self.dataset.get_overlay('mask_idn')

        try:
            usetype_overlay = self.dataset.get_overlay('usetype')
        except dtoolcore.DtoolCoreKeyError:
            if usetype is not None:
                raise
            else:
                pass

        self.im_identifiers = [
            idn for idn in self.dataset.identifiers
            if is_image_overlay[idn]
        ]

        if usetype:
            self.im_identifiers = [
                idn for idn in self.im_identifiers
                if usetype_overlay[idn] == usetype
            ]

        self.mask_identifiers = [
            mask_ids_overlay[idn]
            for idn in self.im_identifiers
        ]

    def __len__(self):
        return len(self.im_identifiers)

    def __getitem__(self, idx):
        image_fpath = self.dataset.item_content_abspath(self.im_identifiers[idx])
        mask_fpath = self.dataset.item_content_abspath(self.mask_identifiers[idx])

        im = Image.from_file(image_fpath)
        mask = Image.from_file(mask_fpath)

        return to_tensor(im), to_tensor(mask)


def image_mask_dataset_from_im_mask_iter(output_base_uri, output_name, im_mask_iter, source_dataset=None):

    if source_dataset is not None:
        image_mask_dataset_from_im_mask_iter_and_dataset(output_base_uri, output_name, im_mask_iter, source_dataset)
    else:
        image_mask_dataset_from_im_mask_iter_only(output_base_uri, output_name, im_mask_iter)


def image_mask_dataset_from_im_mask_iter_only(output_base_uri, output_name, im_mask_iter):

    with dtoolcore.DataSetCreator(
        output_name,
        output_base_uri
    ) as output_ds:
        for n, (image, mask) in enumerate(im_mask_iter):
            metadata_appends = add_image_mask_pair(output_ds, image, mask, n)
            for relpath, key, value in metadata_appends:
                output_ds.add_item_metadata(relpath, key, value)


def image_mask_dataset_from_im_mask_iter_and_dataset(output_base_uri, output_name, im_mask_iter, source_dataset):

    with dtoolcore.DerivedDataSetCreator(
        output_name,
        output_base_uri,
        source_dataset
    ) as output_ds:
        for n, (image, mask) in enumerate(im_mask_iter):
            metadata_appends = add_image_mask_pair(output_ds, image, mask, n)
            for relpath, key, value in metadata_appends:
                output_ds.add_item_metadata(relpath, key, value)


class AIModelDataSetCreator(dtoolcore.DerivedDataSetCreator):

    def __init__(self, output_name, output_base_uri, source_ds):
        try:
            super().__init__(output_name, output_base_uri, source_ds)
        except dtoolcore.storagebroker.StorageBrokerOSError:
            expected_dirpath = os.path.join(output_base_uri, output_name)
            if os.path.isdir(expected_dirpath):
                shutil.rmtree(expected_dirpath)
            super().__init__(output_name, output_base_uri, source_ds)


class LimitDataSetWrapper(torch.utils.data.Dataset):

    def __init__(self, wrapped_dataset, limit=8):
        self.wrapped_dataset = wrapped_dataset
        self.limit = limit

    @property
    def name(self):
        return f"{self.wrapped_dataset.name} (limit {self.limit})"

    @property
    def uuid(self):
        return f"{self.wrapped_dataset.uuid}"

    @property
    def uri(self):
        return f"{self.wrapped_dataset.uri}"

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        return self.wrapped_dataset[idx]
