import click

from dtoolbioimage import Image as dbiImage, scale_to_uint8
from aiutils.unetmodel import unet_model_from_uri

import torch
from torchvision.transforms.functional import to_tensor


def predict_from_unet(model, input_image):

    with torch.no_grad():
        prediction = model(to_tensor(input_image)[None])

    prediction_imarray = prediction.squeeze().numpy()

    return prediction_imarray


@click.command()
@click.argument('image_fpath')
@click.argument('model_uri')
@click.argument('output_fpath')
def main(image_fpath, model_uri, output_fpath):

    model = unet_model_from_uri(model_uri)

    image = dbiImage.from_file(image_fpath)

    prediction = predict_from_unet(model, image)

    prediction.view(dbiImage).save(output_fpath)



if __name__ == "__main__":
    main()
