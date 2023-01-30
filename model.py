import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor


class CAutomaton(nn.Module):

    def __init__(self, num_hidden_states=9,
                       num_hidden_features=96,
                       stochastic=True):
        """
        parameters:
            num_hidden_states = positive int, number of hidden cell-states
            num_hidden_features = positive int, number of channels in hidden layer
            stochastic = False if all cells update according to a global clock
        """
        super().__init__()

        # size of the model
        self.num_states = 3+num_hidden_states
        self.num_hidden_features = num_hidden_features

        # layers
        self.perception_filter = nn.Conv2d(self.num_states,
                                           4*self.num_states,
                                           kernel_size=3,
                                           bias=False)
        self.update_rule = nn.Sequential(nn.Conv2d(4*self.num_states,
                                                   self.num_hidden_features,
                                                   kernel_size=1),
                                         nn.ReLU(),
                                         nn.Conv2d(self.num_hidden_features,
                                                   self.num_states,
                                                   kernel_size=1))
        self.stochastic = stochastic

    def forward(self, x):
        y = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='circular')
        perception = self.perception_filter(y)
        update = self.update_rule(perception)
        if self.stochastic:
            mask = torch.rand(x.size()[-2], x.size()[-1])
            update *= (mask>0.5)
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
    automaton.perception_filter.weight = nn.parameter.Parameter(torch.tensor(kernel),
                                                            requires_grad=False)

    return automaton
