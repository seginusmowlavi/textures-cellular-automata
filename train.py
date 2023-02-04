import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.utils import save_image
from torchvision.transforms import transforms


num_channels = 64
default_channels = {0: 64, 4: 64, 9: 128, 16: 256, 23: 512}
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

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_vggnet_features(device):
    """
    Arguments:
        device = (int or torch.device)
    
    Note: modifies automaton device to match template

    Returns:
        vggnet_features = first half of VGG-16 (convolutions)
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

    if device!='cpu': # this usually means we're on Colab
        # if to be done locally: maybe try
        #    import ssl
        #    ssl._create_default_https_context = ssl._create_unverified_context
        vggnet_features = vgg16(weights=VGG16_Weights.DEFAULT).features
    else:
        vggnet_features = vgg16().features
        vggnet_features.load_state_dict(torch.load('vgg16_features_state_dict.pt'))

    vggnet_features.to(device)
    vggnet_features.eval()
    vggnet_features.requires_grad_(False)

    return vggnet_features


def compute_gram_features(img, vggnet_features,
                          channel_pairs=default_channel_pairs,
                          preprocess=preprocess):
    """
    Compute space-invariant features with outputs of VGG-16 hidden layers.
    More precisely, an image is represented by a sequence of, for each layer,
    a Gram matrix of cross-correlations between channels. The Gram matrices are
    flattened then concatenated into a vector.

    Arguments:
        img             = (tensor) batch of images of size (b, ch, H, W)
        vggnet_features = pretrained VGG-16
        channel_pairs   = (dict of tuples)
                          key   = (int) layer from which to compute features
                          value = (tuple) contains two projection matrices
                                  which reduce the number of channels of layer
                                  vggnet_features[key]
        preprocess      = function applied to imgs before inputting into
                          vggnet_features
    
    Notes:
        - all channel_pairs[key][i] must have same first dimension (ie length)
        - img, vggnet_features and channel_pairs must be on same device

    Returns:
        gram_features = tensor((b, l, m, m)) where l=len(channel_pairs) and
                                                   m=len(channel_pairs[key][i])
    """
    if len(img.shape)==3:
        img = img[np.newaxis, :, :, :]
    b = len(img)
    l = len(channel_pairs)
    m = len(next(iter(channel_pairs.values()))[0])

    gram_features = torch.zeros((b, l, m, m), device=img.device)
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

    return gram_features/gram_features.numel()


def compute_mean_features(img, vggnet_features,
                          channels=default_channels,
                          preprocess=preprocess):
    """
    Compute space-invariant features with outputs of VGG-16 hidden layers.
    More precisely, an image is represented by the means of each channel of
    specified VGG-16 layers.

    Arguments:
        img             = (tensor) batch of images of size (b, ch, H, W)
        vggnet_features = pretrained VGG-16
        channels        = (dict of ints)
                          key   = (int) layer from which to compute features
                          value = (int) size of the layer
        preprocess      = function applied to imgs before inputting into
                          vggnet_features
    
    Notes: img and vggnet_features must be on same device

    Returns:
        mean_features = tensor((b, l)) where l=sum(channels.values())
    """
    if len(img.shape)==3:
        img = img[np.newaxis, :, :, :]
    b = len(img)
    l = sum(channels.values())

    mean_features = torch.zeros((b, l), device=img.device)
    i = 0 # track writer position in mean_features
    
    # pass img through vggnet_features
    out = preprocess(img)
    for idx, layer in enumerate(vggnet_features[:max(channels.keys())+1]):
        out = layer(out)

        if idx in channels:
            b, c, h, w = out.shape

            mean_features[:, i:i+channels[idx]] = out.sum(dim=(2,3))/(h*w)

            i += channels[idx]
    
    return mean_features


def train(automaton, template, step_min=32,
                               step_max=65,
                               batch_dims=(4,128,128),
                               pool_size=128,
                               vgg_mode='gram',
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
        pool_size     = (int) number of states stored in checkpoint pool
        vgg_mode      = 'gram' if we compute gram matrices of vgg channels
                        'mean' if we compute mean of vgg channels
        num_vgg_ch    = (int) if vgg_mode=='gram': number of channels per vgg
                        layer to keep for gram matrix computations
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

    # just in case
    template.requires_grad = False

    # load vgg
    vggnet_features = load_vggnet_features(device)
    
    # initialise optimizer
    optimizer = torch.optim.Adam(automaton.update_rule.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     lr_milestones,
                                                     lr_decay)
    # initialise algorithm
    pool = torch.rand((pool_size, automaton.num_states, h, w), device=device)
    pool_loss = 1e8*np.ones((pool_size)) # empirically, losses are <1e8
    texture_losses, domain_losses, losses = [], [], []
    
    # training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # this loss term ensures that cell states stay in [0,1]
        domain_loss = 0

        # pick states from pool
        batch_idxs = np.random.default_rng().choice(pool_size, b, replace=False)
        idx = np.where(pool_loss==max(pool_loss[batch_idxs]))[0][0]
        pool[idx] = torch.rand((automaton.num_states, h, w), device=device)
        states = pool[batch_idxs]

        # iterate automaton from random initial state
        num_steps = np.random.randint(step_min, step_max)
        for step in range(num_steps):
            states = automaton(states)
            domain_loss += (states - states.clamp(0.0, 1.0)).abs().sum()
        img = states[:, :3, :, :]

        # extract vgg features
        if vgg_mode=='gram':
            channel_pairs = {}
            for key, value in default_channels.items():
                channel_pairs[key] = (nn.functional.normalize(
                               torch.randn((num_vgg_ch, value), device=device)),
                                      nn.functional.normalize(
                               torch.randn((num_vgg_ch, value), device=device)))
            template_fts = compute_gram_features(template, vggnet_features,
                                                 channel_pairs)
            template_fts = torch.cat([template_fts]*b)
            img_fts = compute_gram_features(img, vggnet_features, channel_pairs)
        elif vgg_mode=='mean':
            template_fts = compute_mean_features(template, vggnet_features)
            template_fts = torch.cat([template_fts]*b)
            img_fts = compute_mean_features(img, vggnet_features)

        # compute loss
        if vgg_mode=='gram':
            texture_loss = ((template_fts-img_fts)**2).sum(dim=(1, 2, 3))
        elif vgg_mode=='mean':
            texture_loss = ((template_fts-img_fts)**2).sum(dim=1)
        loss = texture_loss.sum() + domain_loss
        
        # optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        # update pool
        pool[batch_idxs] = states.detach()
        pool_loss[batch_idxs] = texture_loss.tolist()

        # record loss
        texture_losses.append(texture_loss.sum().item())
        domain_losses.append(domain_loss.item())
        losses.append(loss.item())

        # message
        if epoch%100 == 0:
            print(f'Epoch {epoch} complete, loss = {loss.item()}')

    return texture_losses, domain_losses, losses
