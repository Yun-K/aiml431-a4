import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


#TODO:!! Part 2. Finish the following two definition of classes to define Generator and Discriminitor model structure
# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input is the latent vector Z.

        #TODO:!! Define layers here

    def forward(self, x):
        #TODO:!! Define how the data flows in Generator

        return x
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Input Dimension: (nc) x 32 x 32
        #TODO:!! Define layers here


    def forward(self, x):
        #TODO:!! Define how the data flows in Discriminator
        return x
