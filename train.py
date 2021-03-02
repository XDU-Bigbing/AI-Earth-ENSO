import torch
from torch.nn import Linear, LeakyReLU
from networks import net_params, backbone, CausalCNN, ForecastNet, kits


def init_model(device):
    encoder_dccnn = CausalCNN.CausalCNNEncoder(**net_params.dccnn_params)
    encoder = torch.nn.Sequential(backbone.Extractor(), encoder_dccnn)

    regressor_linears = [Linear(**params) for params in net_params.regressor_params]
    i = 1
    while i < len(regressor_linears):
        regressor_linears.insert(i, LeakyReLU())
        i += 2
    regressor_linears.append(kits.squeezeChannels())
    regressor = torch.nn.Sequential(*regressor_linears)

    decoder_linears = [Linear(**params) for params in net_params.decoder_params]
    i = 1
    while i < len(decoder_linears):
        decoder_linears.insert(i, LeakyReLU())
        i += 2
    decoder_linears.append(kits.reshape((-1,net_params.decoder_channels,net_params.decoder_H,net_params.decoder_W)))
    decoder = torch.nn.Sequential(*decoder_linears)

    model = ForecastNet.ForecastNetPlus(
        encoder, regressor, decoder, net_params.sliding_window_size, net_params.output_seq_length, device
    )

    model.to(device)

    return model


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device)

    x = torch.rand((36, 4, 12, 24, 72))
    # x = torch.rand((4, 1))
    # model = 
    y = model(x)
    print(y.size())


if __name__ == "__main__":
    train()
