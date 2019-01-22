import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import get_device


class Normalisation(nn.Module):

    def __init__(self, mean, std):
        super(Normalisation, self).__init__()
        self.mean = torch.Tensor(mean).view(-1, 1, 1)
        self.std = torch.Tensor(std).view(-1, 1, 1)

    def forward(self, image):
        image = (image - self.mean) / self.std
        return image


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
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        input_gram = gram_matrix(x)
        self.loss = F.mse_loss(input_gram, self.target)
        return x


def get_style_model_and_losses(cnn, mean, std, style_image, content_image, style_layers=None, content_layers=None):
    if style_layers is None:
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    if content_layers is None:
        content_layers = ['conv_4']

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

        if name in style_layers:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses


def get_input_optimiser(input_image):
    optimiser = optim.LBFGS([input_image.requires_grad_()])
    return optimiser


def run_style_transfer(cnn,
                       mean,
                       std,
                       input_image,
                       style_image,
                       content_image,
                       n_steps=300,
                       style_weight=1000000,
                       content_weight=1):
    print('Building the style transfer model ...')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, mean, std, style_image, content_image, style_layers=None, content_layers=None
    )
    optimiser = get_input_optimiser(input_image)

    print('Optimising ...')
    run = [0]
    while run[0] <= n_steps:

        def closure():
            input_image.data.clamp_(0, 1)

            optimiser.zero_grad()
            model(input_image)

            style_score = 0
            content_score = 0
            for style_loss in style_losses:
                style_score += style_loss.loss
            for content_loss in content_losses:
                content_score += content_loss.loss

            content_score *= content_weight
            style_score *= style_weight

            loss = content_score + style_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f'run {run}')
                print(f'style loss: {style_score:.6f} - context loss: {content_score:.6f}')
                print('\n')

            return style_score + content_score

        optimiser.step(closure)

    input_image.data.clamp_(0, 1)
    return input_image
