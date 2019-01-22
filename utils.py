import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def show_image(tensor, title=None):
    unloader = transforms.ToPILImage()
    plt.ion()

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

    return None


def test_show_image():
    from model import transform_image_tensors
    style_image, content_image = transform_image_tensors('./data/picasso.jpg', './data/dancing.jpg')
    show_image(style_image)
    return None


if __name__ == '__main__':
    test_show_image()
