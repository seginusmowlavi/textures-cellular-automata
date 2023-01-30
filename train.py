import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from model import CAutomaton

nb_channels = 64
default_channel_pairs = {0:  (torch.eye(nb_channels, 64),
                              torch.eye(nb_channels, 64)),
                         4:  (torch.eye(nb_channels, 64),
                              torch.eye(nb_channels, 64)),
                         9:  (torch.eye(nb_channels, 128),
                              torch.eye(nb_channels, 128)),
                         16: (torch.eye(nb_channels, 256),
                              torch.eye(nb_channels, 256)),
                         23: (torch.eye(nb_channels, 512),
                              torch.eye(nb_channels, 512))}

def load_vggnet_features(colab=False):
    """
    Arguments:
        colab = (bool) False if locally ran, True if on Colab

    Returns:
        vggnet_features = first half of VGG-16 (convolutions)
        preprocess      = preprocessing function
    """
    # net = vgg16() is composed of:
    #     net.features (conv layers)
    #     net.avgpool
    #     net.classifier (fully connected layers)
    # where net.features = nn.Sequential(
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

    pretrained_weights = VGG16_Weights.DEFAULT # DEFAULT = IMAGENET1K_V1
    if colab:
        # if to be done locally: maybe try
        #    import ssl
        #    ssl._create_default_https_context = ssl._create_unverified_context
        vggnet_features = vgg16(weights=pretrained_weights).features
    else:
        vggnet_features = vgg16().features
        vggnet_features.load_state_dict(torch.load('vgg16_features_state_dict.pt'))
    vggnet_features.eval()
    
    preprocess = pretrained_weights.transforms()

    return vggnet_features, preprocess

# DO NOT USE locally with compute_pca=True
# torch.linalg.svd is way too slow for such big matrices!!!
# (first svd computation takes 2min, 10GB on Colab)
# Colab may complain about memory usage
def compute_texture_features_pca(img, vggnet_features, preprocess, pca_size=32, texture_layers=[0, 4, 9, 16, 23], pca_projs=[], compute_pca=True):
    """
    Compute space-invariant features with outputs of VGG-16 hidden layers.
    More precisely, an image is represented by a sequence of, for each layer,
    a Gram matrix of cross-correlations between channels. The Gram matrices are
    flattened then concatenated into a vector. We then compute the L2 loss
    between the vectors for the two images.

    Arguments:
        img             = (tensor) batch of images of size (b, ch, H, W)
        vggnet_features = pretrained VGG-16
        preprocess      = function applied to imgs before inputting into
                          vggnet_features
        pca_size        = (int) number of components in the PCA of hidden layers
        texture_layers  = list of layers from which to compute features
        pca_projs       = (list of tensors) contains the PCA projections of
                          texture layers
                          ie. let (U,S,V) be the SVD of layer texture_layer[i]
                          ( so lines of U are principal directions, lines of
                            U@layer_out are principal components of
                            layer_out=array((ch, h*w)) ),
                          then pca_projs[i, :, :] = U[:pca_size, :]
        compute_pca     = (bool) True if pca_projs=[], in which case pca_projs
                          will be computed

    Returns:
        gram_features = tensor((n, l, pca, pca)) where l=len(texture_layers)
        pca_projs     = same as the input (or computed if compute_svd=True)
        :param img:
        :param vggnet_features:
        :param preprocess:
        :param texture_layers:
        :param pca_size:
        :param compute_svd:
        :param compute_pca:
        :param pca_projs:
    """
    gram_features = torch.zeros((len(img), len(texture_layers), pca_size, pca_size))
    i = 0 # track writer position in gram_features

    # pass img through vggnet_features
    out = preprocess(img)
    for idx, layer in enumerate(vggnet_features[:texture_layers[-1]+1]):
        out = layer(out)

        if idx in texture_layers:
            b, c, h, w = out.shape
            features = out.reshape((b, c, h*w))

            # perform PCA
            if compute_pca:
                feats = features[0] # normally the batch of is size 1
                feats -= torch.mean(feats)
                pca_projs.append(torch.linalg.svd(feats)[0][:pca_size, :])

            # compute features
            pca_comps = pca_projs[i] @ features
            gram_features[:, i, :, :] = pca_comps @ pca_comps.transpose(1,2)

            i += 1

    return gram_features, pca_projs


def compute_texture_features(img, vggnet_features, preprocess,
                             channel_pairs=default_channel_pairs):
    """
    Compute space-invariant features with outputs of VGG-16 hidden layers.
    More precisely, an image is represented by a sequence of, for each layer,
    a Gram matrix of cross-correlations between channels. The Gram matrices are
    flattened then concatenated into a vector. We then compute the L2 loss
    between the vectors for the two images.

    Arguments:
        img             = (tensor) batch of images of size (b, ch, H, W)
        vggnet_features = pretrained VGG-16
        preprocess      = function applied to imgs before inputting into
                          vggnet_features
        channel_pairs   = (dict of tuples)
                          key   = (int) layer from which to compute features
                          value = (tuple) contains two projection matrices
                                  which reduce the number of channels of layer
                                  vggnet_features[key]
    
    Note: all channel_pairs[key][i] must have same first dimension (ie length)

    Returns:
        gram_features = tensor((n, l, m, m)) where l=len(channel_pairs) and
                                                   m=len(channel_pairs[key][i])


    """
    b = len(img)
    l = len(channel_pairs)
    m = len(next(iter(channel_pairs.values()))[0])
    gram_features = torch.zeros((b, l, m, m))
    i = 0 # track writer position in gram_features

    # pass img through vggnet_features
    out = preprocess(img)
    for idx, layer in enumerate(vggnet_features[:max(channel_pairs.keys())+1]):
        out = layer(out)

        if idx in channel_pairs:
            b, c, h, w = out.shape
            features = out.reshape((b, c, h*w))

            # reduce number of channels
            proj1, proj2 = channel_pairs[idx]
            features1, features2 = proj1@features, proj2@features

            # compute space-invariant features (gram matrix)
            gram_features[:, i, :, :] = features1@features2.transpose(1,2)

            i += 1

    return gram_features


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

