"""
Model training and testing of Mirror contrastive loss based sliding window transformer.

Written by Jing Luo from Xi'an University of Technology, China.

luojing@xaut.edu.cn
"""
import logging
import time
import sys
import argparse
import pandas as pd
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from SlidingWinTransformer import SWT
from Margin_loss import *


def train(option, train_set, test_set_existsub, test_set_newsub):
    train_data, train_label = torch.from_numpy(train_set.X), torch.from_numpy(train_set.y)
    test_data_existsub, test_label_existsub = torch.from_numpy(test_set_existsub.X), \
        torch.from_numpy(test_set_existsub.y)
    test_data_newsub, test_label_newsub = torch.from_numpy(test_set_newsub.X), torch.from_numpy(test_set_newsub.y)

    dataset_train = TensorDataset(train_data, train_label)
    dataloader_train = DataLoader(dataset_train, batch_size=option.batch_size, shuffle=True)
    dataset_test = TensorDataset(test_data_existsub, test_label_existsub)
    dataloader_test = DataLoader(dataset_test, batch_size=option.batch_size, shuffle=True)
    dataset_test_2b = TensorDataset(test_data_newsub, test_label_newsub)
    dataloader_test_2b = DataLoader(dataset_test_2b, batch_size=option.batch_size, shuffle=True)

    eeg_size = train_data.shape[-1]
    model = SWT(eeg_size=eeg_size)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=option.lr, weight_decay=0.05)
    lossC = nn.functional.cross_entropy
    # lossDO = losses.MarginLoss()
    lossDO = MarginLoss()
    lossDM = MCL()

    for e in range(option.epochs):
        list_excel = [str(e + 1)]
        train_loss, train_acc = 0.0, 0.0
        model.train()
        # set progress bar
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, (batch_data, batch_label) in pbar:
            batch_data = Variable(batch_data.cuda().type(torch.cuda.FloatTensor))
            batch_label = Variable(batch_label.cuda().type(torch.cuda.LongTensor))
            mirror_data, mirror_label = getMirrorEEG(batch_data, batch_label)

            orig_feature, pred = model(batch_data)
            mirror_feature, pred_temp = model(mirror_data)

            pred_train = torch.max(pred, 1)[1]  # The predicted lable
            train_acc += (pred_train.cpu().numpy() == batch_label.cpu().numpy()).sum()
            if i == len(pbar) - 1:
                length = len(dataset_train)
            else:
                length = (i + 1) * len(batch_label)

            cs_loss = lossC(torch.clamp(pred, min=1e-6), batch_label)
            loss = cs_loss + option.weight_LDO * lossDO(orig_feature, batch_label) + \
                   option.weight_LDM * lossDM(orig_feature, mirror_feature, batch_label, mirror_label)
            train_loss += loss.detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(
                'epoch : {}, train loss: {:.3f}, train acc: {:.3f}'.format(e, train_loss / (i + 1), train_acc / length))
            if length == len(dataset_train):
                list_excel.append(train_loss / (i + 1))
                list_excel.append(train_acc / length)
        model.eval()
        test_acc, mirror_test_acc, test_acc_2b, mirror_test_acc_2b = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            # existing sub
            for batch_data, batch_label in dataloader_test:
                batch_data = Variable(batch_data.cuda().type(torch.cuda.FloatTensor))
                feature, predO = model(batch_data)
                mirror_batch_data, mirror_batch_label = getMirrorEEG(batch_data, batch_label)
                feature, predM = model(mirror_batch_data)
                # porb emsemble
                pred = predM[:, (1, 0)] + predO
                pred_test = torch.max(predO, 1)[1].cpu()
                test_acc += (pred_test == batch_label).sum().numpy()
                pred_test = torch.max(pred, 1)[1].cpu()
                mirror_test_acc += (pred_test == batch_label).sum().numpy()
            test_acc = test_acc / test_data_existsub.shape[0]
            mirror_test_acc = mirror_test_acc / test_data_existsub.shape[0]

            # new sub
            for batch_data, batch_label in dataloader_test_2b:
                batch_data = Variable(batch_data.cuda().type(torch.cuda.FloatTensor))
                feature, predO = model(batch_data)
                mirror_batch_data, mirror_batch_label = getMirrorEEG(batch_data, batch_label)
                feature, predM = model(mirror_batch_data)
                # porb emsemble
                pred = predM[:, (1, 0)] + predO
                pred_test = torch.max(predO, 1)[1].cpu()
                test_acc_2b += (pred_test == batch_label).sum().numpy()
                pred_test = torch.max(pred, 1)[1].cpu()
                mirror_test_acc_2b += (pred_test == batch_label).sum().cpu().numpy()
            test_acc_2b = test_acc_2b / test_data_newsub.shape[0]
            mirror_test_acc_2b = mirror_test_acc_2b / test_data_newsub.shape[0]

            list_excel.append(test_acc)
            list_excel.append(mirror_test_acc)
            list_excel.append(test_acc_2b)
            list_excel.append(mirror_test_acc_2b)
            print('test acc: ', test_acc)
        if e == 0:
            df = pd.DataFrame([['epoch', 'train_loss', 'train_acc', 'test_acc', 'mirror_test_acc', 'test_acc_newsub',
                                'mirror_test_acc_newsub']])  # Column name
            log_data = df.append(pd.DataFrame([list_excel]))
        else:
            log_data = log_data.append(pd.DataFrame([list_excel]))
    log_data.to_excel('./train_2a_' + str(bandpass_low) + 'hz.xlsx', header=None, index=False)


if __name__ == '__main__':
    cuda = True
    bandpass_low = 0
    bandpass_high = 38
    time_start = time.time()
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.DEBUG, stream=sys.stdout)

    data_folder_2a = r'D:\BCI\BCICIV_2a_gdf'
    # data_folder_2a = r'D:\BCI\BCICIV_2a_gdf'
    # data_folder_2b = r'D:\BCI\BCICIV_2b_gdf'
    data_folder_2b = r'D:\BCI\BCICIV_2b_gdf'
    train_set_2a, test_set_2a = read_data_2a(data_folder_2a, bandpass_low, bandpass_high)
    test_set_2b = read_data_2b(data_folder_2b, bandpass_low, bandpass_high)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--weight_LDO', type=float, default=0.2)
    parser.add_argument('--weight_LDM', type=float, default=0.3)
    opt = parser.parse_args()
    train(opt, train_set_2a, test_set_2a, test_set_2b)
