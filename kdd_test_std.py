#-*-coding:utf-8-*-
"""train classifier for NSL-KDD
Created on 2019.3.9

@author: wjj
"""
import argparse
import os
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,f1_score,precision_score,average_precision_score,recall_score
import pandas as pd
import numpy as np
import collections

#-----------------------------costom file
from model.resnet import resnet_s8,resnet_s20,resnet_s56,resnet_s110
from customdataset.KDD import KDD_484, KDD_484_H
from utils.vis_tools import plot_loss_2
#--------------------------------------------

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(5)#固定随机数种子

#arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='kdd_nsl', help='kdd_nsl')
parser.add_argument('--dataroot', default='./data/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=4096, help='input batch size')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.01')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay. default=0.0001')#L2正则项的系数
parser.add_argument('--netD', default='./out/checkpoints/', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args() 
print(opt)
 
# initialization called on ResNet
def weights_init1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or isinstance(m,nn.Linear):
        init.kaiming_normal_(m.weight)


def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 数据转移
device = torch.device("cuda:0")
device_cpu = torch.device("cpu")

# 设置分类器模型
netD = resnet_s8()
res_in_features = netD.fc.in_features
netD.fc = nn.Linear(res_in_features, 2)

netD.to(device)

weights = [1,2]
class_weights = torch.FloatTensor(weights).cuda()
criterion =nn.CrossEntropyLoss() #经过softmax后，计算与target的交叉熵 weight=class_weights
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr,betas=(0.99, 0.999),weight_decay=opt.weight_decay) #beta1 控制动量与当前梯度 beta2 控制梯度平方影响
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, mode='min', factor = 0.9, patience=4, verbose=True, threshold=1)

def test(test_loader):
    netD.eval() #对dropout 和 batch normalization 不一样
    with torch.no_grad():
        correct = 0
        total = 0
        error = 0
        total_label=None
        total_predicted = None
        for i, (image, real_label) in tqdm(enumerate(test_loader)):
            real_cpu = image.to(device, dtype=torch.float)
            label = real_label.to(device, dtype=torch.long)#[128,1]
            outputs = netD(real_cpu)
            errD_real = criterion(outputs, label)

            _, predicted = torch.max(outputs.data, 1)#返回每一行中最大值 及其索引
            total += label.size(0)
            correct += (label == predicted).sum().item()
            #-------------------------------------------------------
            label = label.to(device_cpu).numpy()
            predicted = predicted.to(device_cpu).numpy()   
            if total_label is None:
                total_label = label
                total_predicted = predicted
            else:
                total_label = np.hstack((total_label, label))
                total_predicted = np.hstack((total_predicted, predicted))  

            error += errD_real.item()
            #--------------------------------------------------------
        error_avg = 1.0 * error / (i + 1)

        cm = confusion_matrix(total_label, total_predicted)
        print(cm)

        prec = 1.0 * correct / total
        f = prec
        print('Accuracy of the network on the test images: {:.4f} % ,f score:{:.4f} %'.format(100 * correct / total, 100*f))
        return error_avg, prec, f

def train(train_loader):
    netD.train()
    correct = 0
    total = 0
    error = 0
    for i, (image, real_label) in tqdm(enumerate(train_loader)):
        # train with real
        netD.zero_grad()
        real_cpu = image.to(device, dtype=torch.float)
        label = real_label.to(device, dtype=torch.long)
        output = netD(real_cpu) 
        _, predicted = torch.max(output.data, 1) #返回每一行中最大值 及其索引

        errD_real = criterion(output, label)
        #optimizerD.zero_grad() #in this optimizer,the same as net.zero_grad()
        errD_real.backward()
        optimizerD.step()

        total += label.size(0)
        correct += (label == predicted).sum().item()
        error += errD_real.item()
    error_avg = 1.0 * error / (i+1) #modified by wjj
    prec = 1.0*correct / total
    print('Accuracy of the network on the train image: {} %'.format(100 * correct / total))
    return error_avg

def main(train_loader,test_loader):
    loss_v= []
    acc_v = []
    best_pr = 0

    for epoch in range(opt.niter):
        print('\nepoch: {}'.format(epoch))
        err_d = train(train_loader)

        err_t,prec,pr = test(test_loader)
        loss_v.append(err_t)
        acc_v.append(prec)

        if (len(loss_v) + 1) % 50 == 0:
            plot_loss_2(acc_v, loss_v, epoch + 1,opt.niter, './out/loss_acc/')

        # remember best prec and save checkpoint
        is_best = pr > best_pr
        print("best_pr[%s]",str(best_pr))

        if is_best:
            state = {
                'net':netD.state_dict(),
                'acc':prec,
                'pr':best_pr,
                'epoch':epoch,
                'lr':opt.lr,
                'batchSize':opt.batchSize
            }
            best_pr = max(pr, best_pr)

            torch.save(state, opt.netD + 'model_cnn_resnet110_best.ckpt')
            print(str(epoch)+"_acc[%s]", str(prec))
    # Save the model checkpoint
    torch.save(netD.state_dict(), opt.netD + 'model_cnn_resnet110_1.ckpt')

def load_data():
    print(opt.dataset)

    if opt.dataset == 'kdd_nsl':
        train_path = "./data/KDD_99/train_threshold_194.npy" 
        test_path = "./data/KDD_99/test_threshold_194.npy"
        data_train = np.load(train_path)
        data_test = np.load(test_path)

        x_train = data_train
        x_test = data_test
        train_loader = torch.utils.data.DataLoader(
                                            KDD_484_H(root=opt.dataroot,
                                                    datatrain = x_train,
                                                    datatest = x_test,
                                                    train=True, 
                                                    transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    ]),
                                                    download=True),
                                            batch_size=opt.batchSize, 
                                            shuffle=True)
                                                    

        test_loader = torch.utils.data.DataLoader(
                                            KDD_484_H(root=opt.dataroot,
                                                    datatrain = x_train,
                                                    datatest = x_test,
                                                    train=False, 
                                                    transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    ]),
                                                    download=True),
                                            batch_size=1000, 
                                            shuffle=False)

    return train_loader, test_loader


if __name__=="__main__": 
    start_time = time.time()
    
    train_loader, test_loader = load_data()
    netD.apply(weights_init1)
    main(train_loader, test_loader)
    print("end!")

    end_time = time.time()
    print(end_time - start_time)

