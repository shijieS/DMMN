import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import config

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cuda = config["cuda"]

    def forward(self):
        pass

