import os
import numpy as np
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.nn import Linear, LeakyReLU, MSELoss
from torch.utils.data import DataLoader
import torch.optim as optim
from networks import net_params, backbone, CausalCNN, ForecastNet, SimpleDecoder, kits
from dataHelpers import ENSODataset
from tqdm import tqdm
from numpy import *
import utils, config


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False


def init_model(device):
    encoder_dccnn = CausalCNN.CausalCNNEncoder(**net_params.dccnn_params)
    encoder = torch.nn.Sequential(backbone.Extractor(), encoder_dccnn)

    regressor_linears = [
        Linear(**params) for params in net_params.regressor_params
    ]
    i = 1
    while i < len(regressor_linears):
        regressor_linears.insert(i, LeakyReLU())
        i += 2
    regressor_linears.append(kits.squeezeChannels())
    regressor = torch.nn.Sequential(*regressor_linears)

    decoder = SimpleDecoder.Decoder(**net_params.decoder_params)

    model = ForecastNet.ForecastNetPlus(encoder, regressor, decoder,
                                        net_params.sliding_window_size,
                                        net_params.output_seq_length,
                                        device).double()

    model.to(device)

    return model


def gauss_loss(y_pred, y, sigma=2):
    return 1 - torch.exp(-torch.norm((y_pred - y)) / (2 * sigma))


def test():

    # batch_size = 4
    # epochs = 50

    seed_torch(2021)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device)

    utils.writelog("Model loaded to device")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.8)

    model, start_epoch, optimizer, lr_scheduler = utils.get_checkpoint_state(
        model, optimizer, lr_scheduler)
    model.eval()
    utils.writelog("Model loaded from previous trained")

    path = "tcdata/enso_round1_test_20210201"
    files = os.listdir(path)
    cnt = 0
    for file in files:

        dataset = ENSODataset(file, is_training=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        utils.writelog("Data Loaders created")
        t = tqdm(dataloader, leave=False, total=len(dataloader))

        for i, batch in enumerate(t):

            batch = batch.to(device)

            _, pred_y = model(batch, is_training=False)

            utils.writelog("test {} data over".format(cnt))
            cnt += 1

            if os.path.exists("result"):
                np.save("result/{}".format(file))
            else:
                os.makedirs("result")
                np.save("result/{}".format(file))


if __name__ == "__main__":
    test()
