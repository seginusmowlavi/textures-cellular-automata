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


def compute_texture_features(img, vgg_model, preprocess,
                             texture_layers=[0, 4, 9, 16, 23],
                             pca_size=64
                             pca_projs,
                             compute_svd=True):
    """
    Compute space-invariant features with outputs of VGG-16 hidden layers.
    More precisely, an image is represented by a sequence of, for each layer,
    a Gram matrix of cross-correlations between channels. The Gram matrices are
    flattened then concatenated into a vector. We then compute the L2 loss
    between the vectors for the two images.

    Arguments:
        img            = (array) batch of images of size (b, ch, H, W)
        vgg_model      = pretrained VGG-16
        preprocess     = function applied to imgs before inputting into vgg_model
        pca_size       = (int) number of components in the PCA of hidden layers
        texture_layers = list of layers from which to compute features
        pca_projs      = (list of arrays) contains the PCA projections of texture
                         layers
                         ie. let (U,S,V) be the SVD of layer texture_layer[i]
                         ( so lines of U are principal directions, lines of
                           U@layer_out are principal components of
                           layer_out=array((ch, h*w)) ),
                         then svd[i, :, :] = U[:pca, :]
        compute_pca    = (bool) True if svd = np.zeros(shape), in which case svd
                         will be computed

    Returns:
        gram_features = array((n, l, pca, pca)) where l=len(layers_svd)
        pca_projs     = same as the input (or computed if compute_svd=True)
    """
    b, c, h, w = image.shape
    gram_features = np.zeros((b, len(texture_layers), pca_size, pca_size))
    i = 0 # track writer position in gram_features

    # pass img through vgg_model
    out = preprocess(img)
    for idx, layer in vgg_model.features[:texture_layers[-1]+1]:
        out = layer(out)

        if idx in texture_layers:
            features = out.reshape((b, -1, h*w))

            # perform PCA
            if compute_pca:
                feats = features[0] # normally the batch of is size 1
                feats -= torch.mean(feats)
                pca_projs[i] = torch.linalg.svd(feats)[0][:pca_size, :]

            # compute features
            pca_comps = pca_projs[i] @ features
            gram_features[:, i, :, :] = pca_comps @ pca_comps.transpose((0,2,1))

            i += 1

    return gram_features, pca_projs


def train(automaton, template, rate=[2e-3]*2000+[2e-4]*6000,
                               step_min=32,
                               step_max=64,
                               batch_dims=(4,128,128)):
    """
    Train an automaton

    Arguments:
        automaton  = (CAutomaton) model to train
        template   = single target image
        rate       = list of learning rates (note that #epochs = len(rate))
        (step_min,
         step_max) = range of the number of automaton evolution steps
                     for each training sample

        batch_dims = (tuple (b, h, w)) shape of training batch
    """
    raise NotImplementedError

