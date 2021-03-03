import torch

class unSqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(unSqueezeChannels, self).__init__()

    def forward(self, x):
        return torch.unsqueeze(x, 2)

class squeezeChannels(torch.nn.Module):
    
    def __init__(self):
        super(squeezeChannels, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class reshape(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self, args):
        super(reshape, self).__init__()
        self.args = args
    def forward(self, x):
        return x.view(*self.args)
