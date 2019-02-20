# Adapted from https://github.com/paraschopra/abstract-art-neural-network/blob/master/generate-art.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os, copy
from PIL import Image
import argparse

# valid_activation_choices = ['ELU', 'F', 'GLU', 'Hardshrink', 'Hardtanh', 'LeakyReLU', 'LogSigmoid', 'LogSoftmax', 'Module', 'PReLU', 'Parameter', 'RReLU', 'ReLU', 'ReLU6', 'SELU', 'Sigmoid', 'Softmax', 'Softmax2d', 'Softmin', 'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold']

def get_args():
    parser = argparse.ArgumentParser(description='Use Neural Nets To Generate Pretty Colors')
    parser.add_argument('-x', "--size_x", default=128, dest="size_x", help='x dimension of picture', type=int)
    parser.add_argument('-y', "--size_y", default=128, dest="size_y", help='y dimension of picture', type=int)
    parser.add_argument('-n', "--num_pics", default=1, dest="num_pics", help='number of images to generate', type=int, choices=range(1,20))
    # if more than 20, we get errors: /home/cdleong/miniconda3/envs/art-nn/lib/python3.7/site-packages/matplotlib/pyplot.py:514: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).

    # TODO: option arguments num_neurons, num_layers, activation
    return parser.parse_args()


def init_normal(m):
    if type(m) == nn.Linear:        
        nn.init.normal_(m.weight)


class NN(nn.Module):

    def __init__(self, activation=nn.Tanh, num_neurons=16, num_layers=9):
        """
        num_layers must be at least two
        """
        super(NN, self).__init__()
        self.activation_arg = str(activation)
        layers = [nn.Linear(2, num_neurons, bias=True), activation()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(num_neurons, num_neurons, bias=False), activation()]
        layers += [nn.Linear(num_neurons, 3, bias=False), nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def gen_new_image(size_x, size_y, save=True, **kwargs):
    net = NN(**kwargs)
    net.apply(init_normal)
    colors = run_net(net, size_x, size_y)
    plot_colors(colors)
    if save is True:
        # because bug, must flip order
        prepend="{1}_by_{0}_".format(size_x, size_y)
        save_colors(colors, prepend)
    return net, colors


def run_net(net, size_x=128, size_y=128):
    x = np.arange(0, size_x, 1)
    y = np.arange(0, size_y, 1)
    colors = np.zeros((size_x, size_y, 2))
    for i in x:
        for j in y:
            colors[i][j] = np.array([float(i) / size_y - 0.5, float(j) / size_x - 0.5])
    colors = colors.reshape(size_x * size_y, 2)
    img = net(torch.tensor(colors).type(torch.FloatTensor)).detach().numpy()
    return img.reshape(size_x, size_y, 3)


def plot_colors(colors, fig_size=4):
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(colors, interpolation='nearest', vmin=0, vmax=1)


def save_colors(colors, prepend=""):
    plt.imsave(prepend+str(np.random.randint(100000)) + ".png", colors)


def run_plot_save(net, size_x, size_y, fig_size=8):
    colors = run_net(net, size_x, size_y)
    plot_colors(colors, fig_size)
    save_colors(colors)


if __name__ == "__main__":
    args = get_args()
    
    # there's a  in gen_new_image.
    size_x = args.size_y
    size_y = args.size_x
    for i in range(args.num_pics):
        gen_new_image(size_x, size_y)
