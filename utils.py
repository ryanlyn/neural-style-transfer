import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def tensor_to_image(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def show_image(tensor, title=None):
    image = tensor_to_image(tensor)

    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    return None


def save_image(tensor, path):
    image = tensor_to_image(tensor)
    image.save(path, 'JPEG', quality=90, optimize=True, progressive=True)
    return None

