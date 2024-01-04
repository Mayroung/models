import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
# sys.path.append('/data/mly/CDA/lmmd') 
from models import model, DNET
sys.path.append('/data/mly/CDA') 

from data_pre import load_modulation, load_modulation_lt, load_modulation_SNR_H, load_modulation_SNR_H_lt

def load_data(root_path, src, tar, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    X_train, Y_train, X_test, Y_test, X_SNR_train, X_SNR_test = load_modulation_SNR_H(data='10a')
    NumTrainData = len(X_train)
    print('label distribution of 10a:')
    print('NumTrainData:', NumTrainData)
    print('label distribution of train data:')
    print(list(map(lambda x:sum(Y_train==x),range(11))))
    print('label distribution of test data:')
    print(list(map(lambda x:sum(Y_test==x),range(11))))
    train_set_all = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    test_set_all = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test), torch.tensor(X_SNR_test))
    loader_src = DataLoader(dataset=train_set_all, batch_size=batch_size, shuffle=True,drop_last=True)#unshuffled
    loader_src_test = DataLoader(dataset=test_set_all, batch_size=batch_size, drop_last=False)
    X_train, Y_train, X_test, Y_test, X_SNR_train, X_SNR_test = load_modulation_SNR_H(data='04c')#load_modulation_lt ,imb_ratio=50
    NumTrainData = len(X_train)
    print('label distribution of 04c:')
    print('NumTrainData:', NumTrainData)
    print('label distribution of train data:')
    print(list(map(lambda x:sum(Y_train==x),range(11))))
    print('label distribution of test data:')
    print(list(map(lambda x:sum(Y_test==x),range(11))))
    train_set_all = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    test_set_all = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test), torch.tensor(X_SNR_test))
    loader_tar = DataLoader(dataset=train_set_all, batch_size=batch_size, shuffle=True,drop_last=True)#unshuffled
    loader_tar_test = DataLoader(dataset=test_set_all, batch_size=batch_size, drop_last=False)    
    # loader_src = data_loader.load_training(, src, batch_size, kwargs)
    # loader_tar = data_loader.load_training(, tar, batch_size, kwargs)
    # loader_tar_test = data_loader.load_testing(
    #     root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_src_test, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer):
    model.train()
    source_loader, target_train_loader, _, _ = dataloaders
    datazip = enumerate(zip(source_loader, target_train_loader))
    time_one_epoch_begin = time.perf_counter()
    for step,((data_source, label_source),(data_target, _)) in datazip:
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()
        optimizer.zero_grad()
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1# lambd在训练过程中逐渐增大
        loss = loss_cls + args.weight * lambd * loss_lmmd
        loss.backward()
        optimizer.step()
        if (step+1) % args.log_interval == 0:
            print(
                f'Epoch: [{epoch}], Step:[{step+1}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')
    time_one_epoch_end = time.perf_counter()
    time_one_epoch = (time_one_epoch_end-time_one_epoch_begin)*1000
    time_one_step_avg = time_one_epoch/(step+1)
    print("Epoch{} time_one_epoch:{:.3f}ms time_one_step_avg:{:.3f}ms"
          .format(epoch,                  
                  time_one_epoch,
                  time_one_step_avg))
    time_one_step_avg_list.append(time_one_step_avg) 

def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target,_ in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    model.train()
    return correct

def test_with_snr(model,test_loader_all):
    model.eval()
    correct = 0
    fals = 0
    cmt = torch.zeros(11, 11, 20, dtype=torch.int16)

    for i, data in enumerate(test_loader_all):
        testdata, testlabel, SNR = data
        testdata = torch.reshape(testdata, [-1, 2, 128])
        testdata = testdata.cuda()
        testlabel = testlabel.cuda()
        outputs = model.predict(testdata)
       # _, label = torch.max(testlabel, 1)
        label = testlabel
        _, predict = torch.max(outputs, 1)
        for k in range(len(predict)):
            if predict[k] == label[k]:
                correct = correct + 1
            else:
                fals = fals + 1
            cmt[label[k]][predict[k]][SNR[k]] = cmt[label[k]][predict[k]][SNR[k]] + 1
    model.train()
    acc = correct / (correct + fals)
    #print('Accuracy:',acc)
        
    for j in range(20):
        num = 0
        # print(cmt[:,:,j])
        for k in range(11):
            num = num + cmt[k, k, j]
        #print(int(num) / int(sum(sum(cmt[:, :, j]))))
    print(torch.sum(cmt[:,:,:],dim=2))
    return acc

def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='OFFICE31')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='amazon')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='webcam')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=31)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=300)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.00001, 0.00001, 0.00001])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.005)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=800)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='2')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    time_global_begin = time.perf_counter()
    #global time_one_step_avg_list
    time_one_step_avg_list = []
    time_test_list = []
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    net = DNET().cuda()
    print(net)
    net.eval()
    dataloaders = load_data(args.root_path, args.src,
                            args.tar, args.batch_size)
    acc_src = test_with_snr(net, dataloaders[-2])
    print('acc for src:{:.4f}'.format(acc_src))
    acc_tgt = test_with_snr(net, dataloaders[-1])
    print('acc for src:{:.4f}'.format(acc_tgt))
    max_acc = 0
    stop = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-7)
    for epoch in range(1, args.nepoch + 1):
        train_epoch(epoch, net, dataloaders, optimizer)
        time_test_begin = time.perf_counter()        
        acc = test_with_snr(net, dataloaders[-1])
        time_test_end = time.perf_counter()
        time_test=(time_test_end-time_test_begin)*1000
        time_test_list.append(time_test)       
        if acc > max_acc:
            max_acc = acc
            torch.save(net.state_dict(), 'base_lt.pth')
        print('acc:{:.4f},max acc{:.4f}'.format(acc, max_acc))