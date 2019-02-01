import os
import argparse

import torch
import torchvision.models as models

from gooey import Gooey, GooeyParser

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


def run_and_export(style_path, content_path, output_dir, output_name):
    output = run(style_path, content_path)
    output_path = os.path.join(output_dir, output_name)
    save_image(output, output_path)


@Gooey(program_name='Neural Style Transfer', required_cols=1, optional_cols=1, default_size=(680, 500))
def parse_args_gui():
    parser = GooeyParser(description='neural style transfer between style and content images')

    parser.add_argument(
        'style_path', type=str, widget='FileChooser', help='path to the style image'
    )

    parser.add_argument(
        'content_path', type=str, widget='FileChooser', help='path to the content image'
    )

    parser.add_argument(
        'output_dir', type=str, widget='DirChooser', help='directory in where output should be saved'
    )

    parser.add_argument(
        'output_name', type=str, help='output filename'
    )

    args = parser.parse_args()
    return args


def parse_args():
    parser = argparse.ArgumentParser('neural style transfer between style and content images')

    style_path_default = os.path.join('.', 'data', 'starry_night.jpg')
    parser.add_argument(
        '--style_path', type=str, default=style_path_default, help='path to the style image'
    )

    content_path_default = os.path.join('.', 'data', 'sydney.jpg')
    parser.add_argument(
        '--content_path', type=str, default=content_path_default, help='path to the content image'
    )

    output_dir_default = os.path.join('.', 'output')
    parser.add_argument(
        '--output_dir', type=str, default=output_dir_default, help='directory in where output should be saved'
    )

    output_name_default = 'starry_sydney.jpg'
    parser.add_argument(
        '--output_name', type=str, default=output_name_default, help='output filename'
    )

    args = parser.parse_args()
    return args


def main(args):
    run_and_export(
        style_path=args.style_path,
        content_path=args.content_path,
        output_dir=args.output_dir,
        output_name=args.output_name
    )
    return None


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)

