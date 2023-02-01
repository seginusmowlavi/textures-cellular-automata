import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from train import load_vgg, train
from model import CAutomaton, set_perception_kernels
import argparse
import cv2


if __name__ == '__main__':

    # TODO: à modifier pour fit à notre cas à nous
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_texture', default='images/grid_0008.jpg', required=False, type=str, help='Input texture')
    args = parser.parse_args()

    input_path = args.input_texture

    # Hyperparameters
    mean = 0
    var = 0.1
    sigma = var**0.5

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialisation
    input_texture = Image.open(input_path)
    texture_tensor = preprocess(input_texture)
    texture_batch = texture_tensor.unsqueeze(0)

    img_noise = np.zeros_like(input_texture)
    img_noise = np.random.normal(mean, sigma, img_noise.shape)*255
    img_noise = img_noise.astype(np.uint8)
    noise_tensor = preprocess(Image.fromarray(img_noise))
    noise_batch = noise_tensor.unsqueeze(0)

    # Load models
    pretrained_vgg = load_vgg()
    automaton = CAutomaton()
    automaton = set_perception_kernels(automaton)

    # Feed forward through models
    output_texture_vgg = pretrained_vgg(texture_batch)
    print('Compute texture features OK')

    # TODO: faire passer l'image de bruit dans notre réseau. Problème sur le type de l'image "img_noise" ou dans la
    #  construction du model CAutomaton
    simulated_texture = automaton(img_noise)
    print('Compute simulated noise OK')

    output_simulated_texture_vgg = pretrained_vgg(simulated_texture)
    print('Compute simulated noise vgg output')

    # TODO: Compute the loss of the 2 vgg outputs --> dans le train. Faire les itérations.

    # Save output
    cv2.imwrite('images/noise.jpg', img_noise)
