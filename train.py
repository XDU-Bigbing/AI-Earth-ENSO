import torch
from torch.nn import Linear, LeakyReLU
from networks import net_params, CausalCNN, ForecastNet


def init_model(device):
    encoder_dccnn = CausalCNN.CausalCNNEncoder(**net_params.dccnn_params)
    encoder = None

    regressor_linears = [Linear(**params) for params in net_params.regressor_params]
    i = 1
    while i < len(regressor_linears):
        letters.insert(i, LeakyReLU())
        i += 2
    regressor = torch.nn.Sequential(*regressor_linears)

    decoder = None

    model = ForecastNet.ForecastNetPlus(
        encoder, regressor, decoder, net_params.sliding_window_size, net_params.output_seq_length, device
    )

    model.to(device)

    return model


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device)


if __name__ == "__main__":
    train()
