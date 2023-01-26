import numpy as np

from train import train
from model import CAutomaton
import argparse
import cv2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_texture', required=False, type=str, help='Input texture')
    args = parser.parse_args()

    input_path = args.input_texture

    # Hyperparameters
    mean = 0
    var = 0.1
    sigma = var**0.5

    # Initialisation
    img_texture = cv2.imread(input_path)
    shape = (250, 250, 3)
    img_noise = np.zeros_like(shape)
    img_noise = np.random.normal(mean, sigma, shape)*255
    img_noise = img_noise.astype(np.uint8)

    # Save output
    cv2.imwrite('noise.jpg', img_noise)
