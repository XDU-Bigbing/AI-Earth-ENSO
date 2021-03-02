import torch
import torch.nn as nn

# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.padding = (1, 0, 0)
        self.kernel_size = (3, 3, 3)
        self.dilation = (1, 1, 1)
        self.stride = (1, 1, 1)
        self.activation = nn.LeakyReLU()

        self.conv1 = nn.Conv3d(in_channels=4,
                               out_channels=4,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               stride=self.stride)
        self.conv2 = nn.Conv3d(in_channels=4,
                               out_channels=8,
                               kernel_size=self.kernel_size,
                               padding=(2, 2, 2))
        self.pool = nn.MaxPool3d(kernel_size=(3, 3, 3),
                                  stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(in_channels=8,
                               out_channels=8,
                               padding=self.padding,
                               kernel_size=self.kernel_size,
                               stride=self.stride)
        self.conv4 = nn.Conv3d(in_channels=8,
                               out_channels=16,
                               kernel_size=self.kernel_size,
                               padding=(2, 2, 2))

        self.conv5 = nn.Conv3d(in_channels=16,
                               out_channels=16,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               stride=self.stride)
        self.conv6 = nn.Conv3d(in_channels=16,
                               out_channels=32,
                               kernel_size=self.kernel_size,
                               padding=(2, 2, 2))



    def forward(self, x):
        y = self.conv1(x)
        y = self.activation(y)
        y = self.conv2(y)
        y = self.pool(y)

        y = self.conv3(y)
        y = self.activation(y)
        y = self.conv4(y)
        y = self.activation(y)
        y = self.pool(y)

        y = self.conv5(y)
        y = self.activation(y)
        y = self.conv6(y)
        y = self.activation(y)
        y = self.pool(y)

        return y.view((x.shape[0],-1,x.shape[2]))


