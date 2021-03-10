import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

    # decoder_linears = [Linear(**params) for params in net_params.decoder_params]
    # i = 1
    # while i < len(decoder_linears):
    #     decoder_linears.insert(i, LeakyReLU())
    #     i += 2
    # decoder_linears.append(kits.reshape((-1,net_params.decoder_channels,net_params.decoder_H,net_params.decoder_W)))
    # decoder = torch.nn.Sequential(*decoder_linears)
    decoder = SimpleDecoder.Decoder(**net_params.decoder_params)

    model = ForecastNet.ForecastNetPlus(encoder, regressor, decoder,
                                        net_params.sliding_window_size,
                                        net_params.output_seq_length,
                                        device).double()

    model.to(device)

    return model


def gauss_loss(y_pred, y, sigma=2):
    return 1 - torch.exp(-torch.norm((y_pred - y)) / (2 * sigma))


def train():

    # batch_size = 4
    # epochs = 50

    seed_torch(2021)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = init_model(device)

    print("init model finish")

    utils.writelog("Model loaded to device")

    dataset = ENSODataset('data/soda_train.npy', 'data/soda_label.npy')
    dataloader = DataLoader(dataset,
                            batch_size=config.TRAIN_BATCH_SIZE,
                            shuffle=True)
    utils.writelog("Data Loaders created")

    lossfunc_x = MSELoss().cuda()
    lossfunc_y = MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.8)
    # 如果路径有预训练好的模型，则加载
    if config.IS_CONTINUE:
        model, start_epoch, optimizer, lr_scheduler = utils.get_checkpoint_state(
            model, optimizer, lr_scheduler)
        utils.writelog("Model loaded from previous trained")

    utils.writelog("---------------- Training Started --------------")
    min_loss = 10000000
    for epoch in range(config.EPOCHS):
        t = tqdm(dataloader, leave=False, total=len(dataloader))
        loss_list = []
        for i, (batch, target_x, target_y) in enumerate(t):

            batch = batch.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)

            optimizer.zero_grad()
            model.train()

            pred_x, pred_y = model(batch, is_training=True)
            loss1 = lossfunc_x(pred_x, target_x)
            loss2 = lossfunc_y(pred_y, target_y)
            loss = 0.1 * loss1 + loss2

            utils.writelog("epoch = {}, Loss: {} Loss1: {}, Loss2: {}".format(
                epoch, loss.item(), loss1.item(), loss2.item()))
            t.set_postfix(Loss=loss.item(),
                          Loss1=loss1.item(),
                          Loss2=loss2.item())
            loss_list.append(loss.item())

            # 保存后在更新，否则更新后不一定是最小的
            loss.backward()
            optimizer.step()

        loss_mean = mean(loss_list)
        utils.writelog('========>> Epoch {} : Loss avarage: {}'.format(
            epoch, loss_mean))
        if loss_mean < min_loss:
            min_loss = loss_mean
            utils.save_checkpoint_state(
                epoch, model, optimizer, lr_scheduler,
                config.MODEL_SAVE_PATH)
            utils.writelog(
                ">>>>>>>>>>>>>>>>>>>>> save min loss model in epoch {}<<<<<<<<<<<<<<<<<".format(epoch))


def test():
    seed_torch(2021)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.8)

    load_path = config.MODEL_LOAD_PATH
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    path = "tcdata/enso_round1_test_20210201"
    files = os.listdir(path)
    for file in files:
        data = np.load(file)
        _, pred_y = model(data)
        np.save('result/{}'.format(file), pred_y)


if __name__ == "__main__":
    train()