from PIL import Image

import torch
import torchvision.transforms as transforms

from utils import get_device


def get_image_size():
    image_size = 512 if torch.cuda.is_available() else 256
    return image_size


def get_transforms():
    image_size = get_image_size()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    return transform


def image_loader(image_path, transform):
    device = get_device()

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


def transform_image_tensors(style_image_path, content_image_path):
    transform = get_transforms()

    style_image = image_loader(style_image_path, transform)
    content_image = image_loader(content_image_path, transform)

    assert style_image.size() == content_image.size(), 'style and content images should be the same size'
    return style_image, content_image
