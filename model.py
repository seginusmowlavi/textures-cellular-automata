import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor


class CAutomaton(nn.Module):

    def __init__(self, num_hidden_states=9,
                       num_hidden_features=96,
                       stochastic=True,
                       bias=False,
                       with_first_order=True):
        """
        Parameters:
            num_hidden_states   = (int) number of hidden cell-states
            num_hidden_features = (int) number of channels in hidden layer
            stochastic          = (bool) False if all cells update according to
                                  a global clock
            bias                = (bool) bias of second layer of update_rule
            with_first_order    = (bool) which perception filter to use
                                         True: use identity, 2 Sobels, Laplacian
                                         False: use identity and 3 second-order
                                         operators
        """
        super().__init__()

        # size of the model
        self.num_states = 3+num_hidden_states
        self.num_hidden_features = num_hidden_features

        # perception filter option
        self.with_first_order = with_first_order

        # layers
        self.perception_filter = nn.Conv2d(self.num_states,
                                           4*self.num_states,
                                           kernel_size=3)
        self.bias = bias
        self.update_rule = nn.Sequential(nn.Conv2d(4*self.num_states,
                                                   self.num_hidden_features,
                                                   kernel_size=1),
                                         nn.ReLU(),
                                         nn.Conv2d(self.num_hidden_features,
                                                   self.num_states,
                                                   kernel_size=1,
                                                   bias=self.bias))
        self.stochastic = stochastic

    def forward(self, x):
        y = nn.functional.pad(x, (1, 1, 1, 1), mode='circular')
        perception = self.perception_filter(y)
        update = self.update_rule(perception)
        if self.stochastic:
            mask = torch.randint(0, 2, x.shape, device=x.device)
            update *= mask
        return x+update

def set_perception_kernels(automaton, angle=0):
    """
    Sets the perception filter.
    Since this filter is not learned: requires_grad=False
    """
    n = automaton.num_states
    K = np.zeros((4*n,n,3,3), # shape: (out_ch,in_ch,H,W)
                      dtype=np.float32) # unlike np, torch default is float32

    # identity
    K[np.arange(n),np.arange(n),1,1] = 1

    if automaton.with_first_order:
        # Sobel_x
        Kx = np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]])
        # Sobel_y
        Ky = np.array([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]])
        K[np.arange(n,2*n),np.arange(n),:,:] = np.cos(angle)*Kx+np.sin(angle)*Ky
        K[np.arange(2*n,3*n),np.arange(n),:,:] = np.cos(angle)*Ky-np.sin(angle)*Kx
        # Laplacian
        K[np.arange(3*n,4*n),np.arange(n),:,:] = np.array([[1,  2,1],
                                                           [2,-12,2],
                                                           [1,  2,1]])
    else:
        # d^2/dx^2
        K[np.arange(n,2*n),np.arange(n),:,:] = np.array([[1,-2,1],
                                                         [2,-4,2],
                                                         [1,-2,1]])
        # d^2/dxy
        K[np.arange(2*n,3*n),np.arange(n),:,:] = np.array([[ 1,0,-1],
                                                           [ 0,0, 0],
                                                           [-1,0, 1]])
        # d^2/dy^2
        K[np.arange(3*n,4*n),np.arange(n),:,:] = np.array([[ 1, 2, 1],
                                                           [-2,-4,-2],
                                                           [ 1, 2, 1]])

    automaton.perception_filter.weight = nn.parameter.Parameter(torch.tensor(K))
    automaton.perception_filter.requires_grad_(False)

    return automaton

def initialize_to_zero(automaton):
    """
    Initializes last layer weights to zero
    (Since only one layer is zero: avoids gradient being zero)
    """
    automaton.update_rule[2].weight.data.zero_()
    if automaton.bias:
        automaton.update_rule[2].bias.data.zero_()
    
    return(automaton)