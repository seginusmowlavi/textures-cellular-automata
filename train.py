import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.utils import save_image
from torchvision.transforms import transforms


num_channels = 64
default_channel_pairs = {0:  (torch.eye(num_channels, 64),
                              torch.eye(num_channels, 64)),
                         4:  (torch.eye(num_channels, 64),
                              torch.eye(num_channels, 64)),
                         9:  (torch.eye(num_channels, 128),
                              torch.eye(num_channels, 128)),
                         16: (torch.eye(num_channels, 256),
                              torch.eye(num_channels, 256)),
                         23: (torch.eye(num_channels, 512),
                              torch.eye(num_channels, 512))}

preprocess_bis = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def to_rgb(x):
  return x[...,:3,:,:]+0.5

def load_vggnet_features(device):
    """
    Arguments:
        device = (int or torch.device)
    
    Note: modifies automaton device to match template

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
    if device!='cpu': # this usually means we're on Colab
        # if to be done locally: maybe try
        #    import ssl
        #    ssl._create_default_https_context = ssl._create_unverified_context
        vggnet_features = vgg16(weights=pretrained_weights).features
    else:
        vggnet_features = vgg16().features
        vggnet_features.load_state_dict(torch.load('vgg16_features_state_dict.pt'))

    vggnet_features.to(device)
    vggnet_features.eval()
    vggnet_features.requires_grad_(False)
    
    preprocess = pretrained_weights.transforms()

    return vggnet_features, preprocess


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
    
    Notes:
        - all channel_pairs[key][i] must have same first dimension (ie length)
        - img, vggnet_features and channel_pairs must be on same device

    Returns:
        gram_features = tensor((n, l, m, m)) where l=len(channel_pairs) and
                                                   m=len(channel_pairs[key][i])


    """
    b = len(img)
    l = len(channel_pairs)
    m = len(next(iter(channel_pairs.values()))[0])

    gram_features = torch.zeros((b, l, m, m), device=img.device)
    i = 0 # track writer position in gram_features

    # pass img through vggnet_features
    out = preprocess_bis(img)
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

    return gram_features/gram_features.numel()


def train(automaton, template, step_min=32,
                               step_max=65,
                               batch_dims=(4,128,128),
                               num_vgg_ch=64,
                               num_epochs=8000,
                               lr=1e-3,
                               lr_milestones=[2000],
                               lr_decay=0.1):
    """
    Train an automaton

    Arguments:
        automaton     = (CAutomaton) model to train
        template      = (tensor) single target image
        (step_min,
         step_max)    = (int, int) range of the number of automaton evolution
                        steps for each training sample
                        step_min included, step_max excluded
        batch_dims    = (tuple (b, h, w)) shape of training batch
        num_vgg_ch    = (int) for loss function with vgg: number of channels per
                        vgg layer to keep for gram matrix computations
        num_epochs    = (int) number of training epochs
        lr            = (float) learning rate
        lr_milestones = (list of ints) epochs where learning rate decays
        lr_decay      = (float) learning rate decay
        colab         = (bool) False if locally ran, True if on Colab
    
    Returns:
        losses = (list of floats) history of losses for each epoch
    """
    device = template.device
    automaton.to(device)

    b, h, w = batch_dims
    losses = []

    # just in case
    template.requires_grad = False

    # load vgg
    vggnet_features, preprocess = load_vggnet_features(device)
    
    # initialise optimizer
    optimizer = torch.optim.Adam(automaton.update_rule.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     lr_milestones,
                                                     lr_decay)
    
    # training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()


        # iterate automaton from random initial state
        #states = torch.randn((b, automaton.num_states, h, w), device=device)
        states = torch.rand((b, automaton.num_states, h, w), device=device)
        num_steps = np.random.randint(step_min, step_max)
        for step in range(num_steps):
            states = automaton(states)
        img = states[:, :3, :, :]

        # extract vgg gram features
        channel_pairs = {0: 64, 4: 64, 9: 128, 16: 256, 23: 512}
        for key, value in channel_pairs.items():
            channel_pairs[key] = (nn.functional.normalize(
                               torch.randn((num_vgg_ch, value), device=device)),
                                  nn.functional.normalize(
                               torch.randn((num_vgg_ch, value), device=device)))
        template_fts = compute_texture_features(template[np.newaxis, : , :, :], 
                                                vggnet_features, preprocess,
                                                channel_pairs)
        template_fts = torch.cat([template_fts]*b)
        img_fts = compute_texture_features(img, vggnet_features,
                                           preprocess, channel_pairs)

        # compute loss
        overflow_loss = (states - states.clamp(-1.0, 1.0)).abs().sum()
        loss = torch.sum((template_fts-img_fts)**2) + overflow_loss
        losses.append(loss.item())
        
        # optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        # message
        if epoch%10 == 0:
            print(f'Epoch {epoch} complete, loss = {loss.item()}')

    return losses
