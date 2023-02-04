import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor


class CAutomaton(nn.Module):

    def __init__(self, num_hidden_states=9,
                       num_hidden_features=96,
                       stochastic=True,
                       bias=False):
        """
        parameters:
            num_hidden_states   = (int) number of hidden cell-states
            num_hidden_features = (int) number of channels in hidden layer
            stochastic          = (bool) False if all cells update according to
                                  a global clock
            bias                = (bool) bias of second layer of update_rule
        """
        super().__init__()

        # size of the model
        self.num_states = 3+num_hidden_states
        self.num_hidden_features = num_hidden_features

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

def set_perception_kernels(automaton):
    """
    Sets the perception filter to concat(identity,sobels,laplacian)
    Since this filter is not learned: requires_grad=False
    """
    n = automaton.num_states
    kernel = np.zeros((4*n,n,3,3), # shape: (out_ch,in_ch,H,W)
                      dtype=np.float32) # unlike np, torch default is float32
    kernel[np.arange(n),np.arange(n),1,1] = 1 # first channels: cell-state
    kernel[np.arange(n,2*n),np.arange(n),:,:] = np.array([[-1,0,1],
                                                          [-2,0,2],
                                                          [-1,0,1]]) # Kx
    kernel[np.arange(2*n,3*n),np.arange(n),:,:] = np.array([[-1,-2,-1],
                                                            [ 0, 0, 0],
                                                            [ 1, 2, 1]]) # Ky
    kernel[np.arange(3*n,4*n),np.arange(n),:,:] = np.array([[1,  2,1],
                                                            [2,-12,2],
                                                            [1,  2,1]]) # Klap
    automaton.perception_filter.weight = nn.parameter.Parameter(torch.tensor(kernel))
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