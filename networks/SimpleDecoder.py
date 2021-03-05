import torch
from kits import unSqueezeChannels, reshape, flatten

class Decoder(torch.nn.Module):

    def __init__(self, in_channels=320, reduced_size = 160, input_dim = 4, upsample_size_one = (12, 36), upsample_size_two = (24, 72), kernel_size=3, stride=1, padding=1, out_padding=0):
        super(Decoder, self).__init__()

        #(B, D) -> (B, C2, 12)   ========   (B, 320) -> (B, 512, 12)
        self.linear_1 = torch.nn.Linear(in_channels, reduced_size) #(B, 320) -> (B, 160)
        self.relu1 = torch.nn.LeakyReLU()
        self.unsqueeze_2 = unSqueezeChannels(2) #(B, 160) -> (B, 160, 1)
        self.unsqueeze_3 = unSqueezeChannels(3)  # (B, 160, 1) -> (B, 160, 1, 1)
        self.unsampling = torch.nn.Upsample(size=upsample_size_one) #(B, 160, 1, 1)  ->  (B, 160, 12, 36)
        self.convT_1 = torch.nn.ConvTranspose2d(reduced_size, input_dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding)#(B, 160, 12, 36)  ->  (B, 4, 12, 36)
        self.flatten_1 = flatten()
        self.linear_2 = torch.nn.Linear(input_dim * upsample_size_one[0] * upsample_size_one[1], input_dim * upsample_size_two[0] * upsample_size_two[1])  # (B, 320) -> (B, 160)
        self.relu2 = torch.nn.LeakyReLU()
        self.reshape = reshape((-1, input_dim, upsample_size_two[0], upsample_size_two[1]))
        self.convT_2 = torch.nn.ConvTranspose2d(input_dim, input_dim, kernel_size=kernel_size, stride=stride,padding=padding, output_padding=out_padding)  # (B, 160, 12, 36)  ->  (B, 4, 12, 36)


    def forward(self, inputs):
        hidden = self.linear_1(inputs)
        hidden = self.relu1(hidden)
        hidden = self.unsqueeze_2(hidden)
        hidden = self.unsqueeze_3(hidden)
        hidden = self.unsampling(hidden)
        hidden = self.convT_1(hidden)
        hidden = self.flatten_1(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.reshape(hidden)
        output = self.convT_2(hidden)

        return output

