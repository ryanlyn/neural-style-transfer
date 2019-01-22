import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from utils import get_device, show_image


def get_image_size():
    image_size = 512 if torch.cuda.is_available() else 128
    return image_size


def get_transforms():
    transform = transforms.Compose([
        transforms.Resize(get_image_size()),
        transforms.ToTensor()
    ])
    return transform


def image_loader(image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image


def transform_image_tensors(content_image_path, style_image_path):
    transform = get_transforms()

    content_image = image_loader(content_image_path, transform)
    style_image = image_loader(style_image_path, transform)

    assert style_image.size() == content_image.size(), 'style and content images should be the same size'
    return content_image, style_image


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


def gram_matrix(x):
    # a = batch_size
    # b = number of feature maps
    # c, d = dimensions of feature maps
    a, b, c, d = x.size()

    features = x.view(a * b, c * d)
    gram = torch.mm(features, features.t())

    # we normalise the values of the gram matrix by dividing by the number of elements in each feature map
    gram = gram.div(a * b * c * d)
    return gram


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target_gram = gram_matrix(target_feature).detach()

    def forward(self, x):
        input_gram = gram_matrix(x)
        self.loss = F.mse_loss(input_gram, self.target_gram)
        return x


class Normalisation(nn.Module):
    def __init__(self, mean, std):
        super(Normalisation, self).__init__()
        self.mean = torch.Tensor(mean).view(-1, 1, 1)
        self.std = torch.Tensor(std).view(-1, 1, 1)

    def forward(self, image):
        image = (image - self.mean) / self.std
        return image


def get_style_model_and_losses(cnn, mean, std, style_image, content_image, content_layers=None, style_layers=None):
    if content_layers is None:
        content_layers = ['conv_4']
    if style_layers is None:
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    device = get_device()
    cnn = copy.deepcopy(cnn)

    normalisation = Normalisation(mean, std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalisation)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognised layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, content_losses, style_losses


def get_input_optimiser(input_image):
    optimiser = optim.LBFGS([input_image.requires_grad_()])
    return optimiser


def run_style_transfer(cnn,
                       mean,
                       std,
                       content_image,
                       style_image,
                       input_image,
                       n_steps=300,
                       style_weight=1000000,
                       content_weight=1):
    print('Building the style transfer model ...')
    model, content_losses, style_losses = get_style_model_and_losses(
        cnn, mean, std, style_image, content_image, content_layers=None, style_layers=None
    )
    optimiser = get_input_optimiser(input_image)

    print('Optimising ...')
    run = [0]
    while run[0] < n_steps:
        def closure():
            imput_image = input_image.data.clamp(0, 1)

            optimiser.zero_grad()
            model(input_image)

            content_score = 0
            style_score = 0
            for content_loss in content_losses:
                content_score += content_loss
            for style_loss in style_losses:
                style_score += style_loss

            content_score *= content_weight
            style_score *= style_weight

            loss = content_score + style_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f'run {run}')
                print(f'style loss: {style_score:.6f} - context loss: {content_score:.6f}')
                print('\n')

            return content_score + style_score

        optimiser.step(closure)

    input_image = input_image.data.clamp(0, 1)
    return input_image


if __name__ == '__main__':
    device = get_device()
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    cnn_std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    content_image, style_image = transform_image_tensors('./data/dancing.jpg', './data/picasso.jpg')
    output = run_style_transfer(
        cnn,
        cnn_mean,
        cnn_std,
        content_image,
        style_image,
        input_image=content_image,
        n_steps=300,
        style_weight=1000000,
        content_weight=1
    )
    show_image(output)
