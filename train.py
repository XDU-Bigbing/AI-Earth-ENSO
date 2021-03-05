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

    regressor_linears = [Linear(**params) for params in net_params.regressor_params]
    i = 1
    while i < len(regressor_linears):
        regressor_linears.insert(i, LeakyReLU())
        i += 2
    regressor_linears.append(kits.squeezeChannels())
    regressor = torch.nn.Sequential(*regressor_linears)

    # decoder_linears = [Linear(**params) for params in net_params.decoder_params]
    # i = 1
    # while i < len(decoder_linears):
    #     decoder_linears.insert(i, LeakyReLU())
    #     i += 2
    # decoder_linears.append(kits.reshape((-1,net_params.decoder_channels,net_params.decoder_H,net_params.decoder_W)))
    # decoder = torch.nn.Sequential(*decoder_linears)
    decoder = SimpleDecoder.Decoder(**net_params.decoder_params)

    model = ForecastNet.ForecastNetPlus(
        encoder, regressor, decoder, net_params.sliding_window_size, net_params.output_seq_length, device
    ).double()

    model.to(device)

    return model

def gauss_loss(y_pred, y, sigma=2):
    return 1-torch.exp(-torch.norm((y_pred-y)) / (2 * sigma ))

def train():

    batch_size = 4
    epochs = 50

    seed_torch(2021)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device)

    dataset = ENSODataset('/content/gdrive/MyDrive/SODA_DATA/soda_train.npy','/content/gdrive/MyDrive/SODA_DATA/soda_label.npy')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)    

    
    lossfunc_x = MSELoss().cuda()
    lossfunc_y = MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        t = tqdm(dataloader, leave=False, total=len(dataloader))
        loss_list = []
        for i, (batch, target_x,target_y) in enumerate(t):
            
            batch = batch.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)

            optimizer.zero_grad()
            model.train()

            pred_x, pred_y = model(batch)
            loss1 = lossfunc_x(pred_x, target_x)
            loss2 = lossfunc_y(pred_y, target_y)
            loss = 0.1*loss1+loss2

            # print('Loss: {} Loss1: {}, Loss2: {}'.format(loss.item(),loss1.item(),loss2.item()))
            t.set_postfix(Loss=loss.item(),Loss1=loss1.item(),Loss2=loss2.item())
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()

        print('Loss avarage:',mean(loss_list))

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device)

    x = torch.rand((2, 4, 12, 24, 72))
    # x = torch.rand((4, 1))
    # model = 
    y = model(x)
    print(y.size())

if __name__ == "__main__":
    train()
