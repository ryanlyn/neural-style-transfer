import os

import torch
import torchvision.models as models

from model import run_style_transfer
from loader import transform_image_tensors
from utils import get_device, save_image


def run(style_path, content_path):
    device = get_device()
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    cnn_std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    style_image, content_image = transform_image_tensors(style_path, content_path)
    input_image = content_image.clone()

    output = run_style_transfer(
        cnn=cnn,
        mean=cnn_mean,
        std=cnn_std,
        input_image=input_image,
        style_image=style_image,
        content_image=content_image,
        n_steps=300,
        style_weight=1000000,
        content_weight=1
    )
    return output


def run_and_export(style_path, content_path, output_path):
    output = run(style_path, content_path)
    save_image(output, output_path)


if __name__ == '__main__':
    style_image_path = os.path.join('.', 'data', 'picasso.jpg')
    content_image_path = os.path.join('.', 'data', 'dancing.jpg')
    output_image_path = os.path.join('.', 'output', 'picasso-dancing.jpg')
    run_and_export(style_path=style_image_path, content_path=content_image_path, output_path=output_image_path)

