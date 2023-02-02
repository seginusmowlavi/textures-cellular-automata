import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from model import *
from train import *


if __name__ == '__main__':

    # Check if cuda available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    templatePIL = plt.imread('images/bubbles.jpg')
    template = ToTensor()(templatePIL).to(device)

    plt.imshow(templatePIL)
    plt.axis('off')
    plt.show()

    automaton = CAutomaton()
    set_perception_kernels(automaton)
    initialize_to_zero(automaton)

    automaton.to(device)

    losses = train(automaton, template, num_epochs=1000)

    print(automaton)
