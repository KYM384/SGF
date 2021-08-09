from torch.nn import functional as F
from torch import nn
import torch


def AdaIN(x, c):
    m_x = torch.mean(x, 1, keepdim=True)
    s_x = torch.std(x, 1, keepdim=True)
    x_norm = (x - m_x) / (s_x + 1e-5)
    m_c = torch.mean(c, 1, keepdim=True)
    s_c = torch.std(c, 1, keepdim=True)
    return x_norm * s_c + m_c


class ConditionalLinearBlock(nn.Module):
    def __init__(self, out_ch, c_dim):
        super().__init__()
        self.fc1 = nn.utils.spectral_norm(nn.Linear(c_dim, out_ch))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(out_ch, out_ch))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, c):
        c = self.fc1(c)
        h = AdaIN(x, c)
        h = self.lrelu(h)
        h = self.fc2(h)
        return h


class AuxiliaryMappingNetwork(nn.Module):
    def __init__(self, n_layer, c_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append( ConditionalLinearBlock(512, c_dim) )

    def forward(self, x, c):
        h = x
        for layer in self.layers:
            h = layer(h, c)
        return h