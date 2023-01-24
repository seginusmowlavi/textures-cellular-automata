import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from model import CAutomaton


def load_vgg():
    """
    Returns:
        vgg_model = pretrained VGG-16
        preprocess = function applied to images before inputting into vgg_model
    """
    pretrained_weights = VGG16_Weights.DEFAULT # DEFAULT = IMAGENET1K_V1
    vgg_model = vgg16(weights=pretrained_weights)
    # vgg_model is vgg_model.features (conv layers)
    #              followed by a classifier (avgpool then 3 FC layers)
    # vgg_model.features = nn.Sequential(
    #     nn.Conv2d(  3,  64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.Conv2d( 64,  64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d( 64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=2, stride=2) )
    vgg_model.eval()
    preprocess = pretrained_weights.transforms()
    return vgg_model, preprocess


def compute_texture_features(img, vgg_model, preprocess, pca=None):
    """
    Compute space-invariant features with outputs of VGG-16 hidden layers.
    More precisely, an image is represented by a sequence of, for each layer,
    a Gram matrix of cross-correlations between channels. The Gram matrices are
    flattened then concatenated into a vector. We then compute the L2 loss
    between the vectors for the two images.

    Arguments:
        img = batch of images of size (ch,H,W)
        vgg_model = pretrained VGG-16
        preprocess = function applied to images before inputting into vgg_model
        pca = list of tuples (S,V) where for each layer giving features,
              (U,S,V) are the svd outputs for the PCA on the channels of the
              layer
              If pca=None: the PCA is computed
    """
    # pass img through vgg_model
    for idx, layer in vgg_model.features.named_children():
        feat = layer(feat)
        if some condition on idx:
            TODO


def texture_loss(img1, img2, vgg_model, preprocess):
    """
    TO DO: rewrite the whole function
    """
    # preprocess the inputs
    assert img1.size() == img2.size()
    img1, img2 = preprocess(img1), preprocess(img2)


def train(automaton, template, rate=[2e-3]*2000+[2e-4]*6000,
                               step_min=32,
                               step_max=64):
    """
    Train an automaton with the following parameters:
        template = image
        rate = list of learning rates (note that #epochs = len(rate))
        (step_min,step_max) = range of the number of automaton evolution steps
                              for each training sample
    """
    raise NotImplementedError
