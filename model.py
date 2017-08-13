import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init

class UNet(nn.Module):
    """ UNet based on https://arxiv.org/abs/1505.04597
    """

    def __init__(self):
        pass

    def forward(self, x):
        pass