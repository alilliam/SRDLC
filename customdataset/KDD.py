# -*- coding: utf-8 -*-
""" make dataset
"""
from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from scipy import ndimage
import torch
import matplotlib.pyplot as plt
import collections


class KDD_122f(data.Dataset):
    """
        transform 41d to 122d
    """

    base_folder = 'KDD_99_old'
    train_txt = "KDDTrain+_value.txt"
    test_txt = "KDDTest+_value.txt"
    r2l_1 = "new_r2l_data_41d_[1].npy"
    r2l_2 = "new_r2l_data_41d_[2].npy"
    r2l_3 = "new_r2l_data_41d_[2_1].npy"
    r2l_4 = "new_r2l_data_41d_[32].npy"
    r2l_5 = "new_r2l_data_41d_[33-35]_z0_v2.npy"


    def divide_by_tens(self,data_int):
        cnt = 0
        while(data_int >= 1):
            cnt += 1
            data_int //= 10
        return int(cnt)

    def rounding(self,data):
        divide_by_tens_ufunc = np.frompyfunc(self.divide_by_tens,1,1)
        data = divide_by_tens_ufunc(data)
        return data

    def __init__(self, root):
        self.root = os.path.expanduser(root)
        self.data = []
        self.targets = []

        file_path_train = os.path.join(self.root, self.base_folder, self.train_txt)
        file_path_test = os.path.join(self.root, self.base_folder, self.test_txt)

        #-------------------------
        data_info_train = pd.read_csv(file_path_train,header=None)
        data_info_test = pd.read_csv(file_path_test,header=None)

        #------------------------------------------------
        # r2l_1_path = os.path.join(self.root, self.base_folder, self.r2l_1)
        # r2l_2_path = os.path.join(self.root, self.base_folder, self.r2l_2)
        # r2l_3_path = os.path.join(self.root, self.base_folder, self.r2l_3)
        # r2l_4_path = os.path.join(self.root, self.base_folder, self.r2l_4)
        r2l_5_path = os.path.join(self.root, self.base_folder, self.r2l_5)


        # r2l_1 = np.load(r2l_1_path)
        # r2l_2 = np.load(r2l_2_path)
        # r2l_3 = np.load(r2l_3_path)
        # r2l_4 = np.load(r2l_4_path)
        r2l_5 = np.load(r2l_5_path)

        # data_info_test = np.vstack((r2l_1,r2l_2,r2l_3,r2l_4))
        # np.save("new_r2l_41d.npy",data_info_test)
        # data_info_r2l = np.load("new_r2l_41d.npy")
        data_info_r2l = r2l_5

        #------------------------------------------------

        data_train,flag_train = self.get_body(data_info_train)
        data_test,flag_test = self.get_body(data_info_test)
        data_r2l,flag_r2l = self.get_body1(data_info_r2l)

        data_train = np.vstack((data_train,data_r2l))
        flag_train = np.vstack((flag_train,flag_r2l))

        min_max_scalar = preprocessing.MinMaxScaler()
        X_train_minmax1 = min_max_scalar.fit_transform(data_train)
        X_plus_flag1 = np.hstack((X_train_minmax1,flag_train))

        X_train_minmax2 = min_max_scalar.transform(data_test)
        X_train_minmax2[X_train_minmax2 > 1] = 1
        X_train_minmax2[X_train_minmax2 < 0] = 0
        X_plus_flag2 = np.hstack((X_train_minmax2,flag_test))

        X_train_minmax3 = min_max_scalar.transform(data_r2l)
        X_train_minmax3[X_train_minmax3 > 1] = 1
        X_train_minmax3[X_train_minmax3 < 0] = 0
        X_plus_flag3 = np.hstack((X_train_minmax3,flag_r2l))
        # np.save("../data/KDD_99/a_v2.npy",X_plus_flag1)
        # np.save("../data/KDD_99/b_v2.npy",X_plus_flag2)
        # np.save("../data/KDD_99/2_v2_r2l_[33-35]_z0_v2.npy",X_plus_flag3)

    def get_body(self, data_info):
        # x1 = np.asarray(data_info.iloc[:,1])
        # nb_classes1 = 3
        # targets1 = x1.reshape(-1)
        # one_hot_targets1 = np.eye(nb_classes1)[targets1-1]

        # y1 = np.asarray(data_info.iloc[:,2])
        # y1[y1>25] -= 1
        # nb_classes2 = 70
        # targets2 = y1.reshape(-1)
        # one_hot_targets2 = np.eye(nb_classes2)[targets2-1]

        # z1 = np.asarray(data_info.iloc[:,3])
        # nb_classes3 = 11
        # targets3 = z1.reshape(-1)
        # one_hot_targets3 = np.eye(nb_classes3)[targets3-1]

        # m1 = np.asarray(self.rounding(data_info.iloc[:,4]),dtype=np.uint8)#data_v2 feature 5,6 rounding
        # nb_classes4 = 11
        # targets4 = m1.reshape(-1)
        # one_hot_targets4 = np.eye(nb_classes4)[targets4]

        # n1 = np.asarray(self.rounding(data_info.iloc[:,5]),dtype=np.uint8)
        # nb_classes5 = 11
        # targets5 = n1.reshape(-1)
        # one_hot_targets5 = np.eye(nb_classes5)[targets5]

        flag1 = np.asarray(data_info.iloc[:,41]).reshape(-1,1)
        body1 = np.asarray(data_info)
        body1[:,4] = np.asarray(self.rounding(data_info.iloc[:,4]),dtype=np.uint8)#data_v2 feature 5,6 rounding
        body1[:,5] = np.asarray(self.rounding(data_info.iloc[:,5]),dtype=np.uint8)
        data1 = np.delete(body1,[41,42],axis=1)#[1,2,3,41,42]

        #-----------------------
        # temp_delete1 = np.delete(body1,[1,2,3,4,5,41,42],axis=1)#[1,2,3,41,42]
        # data1 = np.hstack((temp_delete1,one_hot_targets1,one_hot_targets2,one_hot_targets3,one_hot_targets4,one_hot_targets5))
        return data1,flag1

    def get_body1(self, data_info):
        # x1 = np.asarray(data_info[:,1],dtype=np.uint8)
        # nb_classes1 = 3
        # targets1 = x1.reshape(-1)
        # one_hot_targets1 = np.eye(nb_classes1)[targets1-1]

        # y1 = np.asarray(data_info[:,2],dtype=np.uint8)
        # y1[y1>25] -= 1
        # nb_classes2 = 70
        # targets2 = y1.reshape(-1)
        # one_hot_targets2 = np.eye(nb_classes2)[targets2-1]

        # z1 = np.asarray(data_info[:,3],dtype=np.uint8)
        # nb_classes3 = 11
        # targets3 = z1.reshape(-1)
        # one_hot_targets3 = np.eye(nb_classes3)[targets3-1]

        # m1 = np.asarray(self.rounding(data_info[:,4]),dtype=np.uint8)#data_v2 feature 5,6 rounding
        # nb_classes4 = 11
        # targets4 = m1.reshape(-1)
        # one_hot_targets4 = np.eye(nb_classes4)[targets4]

        # n1 = np.asarray(self.rounding(data_info[:,5]),dtype=np.uint8)
        # nb_classes5 = 11
        # targets5 = n1.reshape(-1)
        # one_hot_targets5 = np.eye(nb_classes5)[targets5]

        flag1 = np.asarray(data_info[:,41]).reshape(-1,1)
        body1 = np.asarray(data_info)
        body1[:,4] = np.asarray(self.rounding(data_info[:,4]),dtype=np.uint8)#data_v2 feature 5,6 rounding
        body1[:,5] = np.asarray(self.rounding(data_info[:,5]),dtype=np.uint8)

        data1 = np.delete(body1,[41,42],axis=1)#[1,2,3,41,42]
        #-----------------------
        # temp_delete1 = np.delete(body1,[1,2,3,4,5,41,42],axis=1)#[1,2,3,41,42]
        # data1 = np.hstack((temp_delete1,one_hot_targets1,one_hot_targets2,one_hot_targets3,one_hot_targets4,one_hot_targets5))
        return data1,flag1
  

class KDD_484(data.Dataset):
    '''
        attackType_rate: The map of type_attack and the rate in the data besides the attackType
        type_attack : 
                      1--normal
                      2--probe
                      3--r2l&u2r
                      5--DoS1
                      6--DoS2
                      7--DoS3
    '''
    attackType_rate = {1:0.9,2:0.102,3:0.0244,5:0.0579,6:0.375,7:0.0385}#0.9|0.87,0.273 0.0346
    base_folder = 'KDD_99'
    y_train_file_name ="y_pred.npy"
    # r2l_2000 = "new_r2l_41d.npy"

    def __init__(self, root, datatrain,datatest,type_attack,train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.data = []
        self.targets = []
        self.type_attack = type_attack

        #---------------------------------------------------------------------
        # test_data =np.load(file_path_test)#所有训练集
        # train1 = np.load(file_path_train)
        # data_all_1 = np.vstack((train1,test_data))
        # data_all,test =train_test_split(data_all_1,test_size = 0.2,random_state=5)#change!0.1518
        #---------------------------------------------------------------------
        data_all = datatrain#所有训练集
        _,w = data_all.shape
        r2l = None

        if self.train:
            if ((self.type_attack == 5) |(self.type_attack == 6)|(self.type_attack == 7)):
                file_path1 = os.path.join(self.root, self.base_folder, self.y_train_file_name)
        

        if self.train:
            arr_1 = data_all[:,w-1]

            #确定训练集
            if self.type_attack == 3:
                r2l = data_all[(arr_1==3) | (arr_1==4)].copy()
                temp2 = data_all[(arr_1!=3) &(arr_1!=4)]
            elif ((self.type_attack == 5) |(self.type_attack == 6)|(self.type_attack == 7)):
                flag =np.load(file_path1).reshape(-1)
                flag1 = data_all[:,w-1].reshape(-1)
                flag1[flag1==0] = flag

                r2l = data_all[flag1==self.type_attack].copy()
                temp2 = data_all[flag1!=self.type_attack]
            else:
                r2l = data_all[arr_1==self.type_attack].copy()
                temp2 = data_all[arr_1!=self.type_attack]

            if self.type_attack !=1:#train_test_split test_size 为1时不代表数据占比为1
                a_train,a_test =train_test_split(temp2,test_size = self.attackType_rate[self.type_attack],random_state=3)#change! self.attackType_rate[self.type_attack]
            else:
                a_train,r2l =train_test_split(r2l,test_size = 0.87,random_state=3)#change! self.attackType_rate[self.type_attack]
                a_test = temp2

            print(len(r2l))
            print(len(a_test))
            arr_negative= np.zeros(len(a_test))
            arr_positive = np.ones(len(r2l))
            arr = np.hstack((arr_positive,arr_negative)).reshape(-1,1)
            temp_mix = np.vstack((r2l,a_test))
            

        else:
            #测试集
            temp_mix = datatest
            arr_1 = temp_mix[:,w-1]
            arr = arr_1.copy()
            
            #打标签
            if ((self.type_attack == 5) |(self.type_attack == 6)|(self.type_attack == 7)):
                arr[(arr_1==0)] = 1
                arr[(arr_1!=0)] = 0
            elif self.type_attack !=3:
                arr[(arr_1==self.type_attack)] = 1
                arr[(arr_1!=self.type_attack)] = 0
            else:
                arr[(arr_1==3)| (arr_1==4)] = 1
                arr[(arr_1!=3) &(arr_1!=4)] = 0
        #//////////////////////////////////////////////
        #------------------------------------均值
        temp_mean = np.hsplit(data_all,np.array([w-1,w+1]))#data_all train1
        arr_t_mean = temp_mean[1].copy().reshape(-1)
        arr_t_mean[arr_t_mean!=1]=0

        temp_121_mean = temp_mean[0]
        # temp_121_mean = np.delete(temp_mean[0],[6],axis=1)

        positive = temp_121_mean[arr_t_mean==1]
        mean_pos = np.mean(positive,axis = 0)

        negtive = temp_121_mean[arr_t_mean==0]
        mean_neg = np.mean(negtive,axis = 0)
        #----------------------------------------------
        temp = np.hsplit(temp_mix,np.array([w-1,w+1]))
        arr_t = temp[1].copy().reshape(-1)
        arr_t[arr_t!=1]=0

        temp_121 = temp[0]
        # temp_121 = np.delete(temp[0],[6],axis=1)

        temp_121_1_t = temp_121-mean_pos
        temp_121_1 = np.abs(temp_121_1_t)
        temp_121_1[temp_121_1>0.5]=1#差异大于0.5的高亮
        temp_121_1[temp_121_1<=0.5]=0

        temp_121_2_t = temp_121-mean_neg
        temp_121_2 = np.abs(temp_121_2_t)
        temp_121_2[temp_121_2>0.5]=1
        temp_121_2[temp_121_2<=0.5]=0

        temp_121_3 = temp_121*temp_121

        #////////////////////////////////////////////////
        #40d
        # temp_121_1 = temp_121.copy()
        # temp_121_2 = temp_121.copy()
        # temp_121_3 = temp_121.copy()
        #////////////////////////////////////////////////

        temp_t = np.hstack((temp_121,temp_121_1,temp_121_2,temp_121_3))
        temp_data = temp_t[:,:676]#为了保证特征形状
        
        self.data = temp_data
        self.targets = arr

        #///////////////////////////////////////////////
        self.data = self.data.reshape(-1, 1, 26, 26)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        '''
        k = np.array([[1,2,1],
        [2,4,2],
        [1,2,1]])
        
        k = k/k.sum()
        img2 =img.squeeze(0)
        img1 =ndimage.convolve(img2,k)
        img3 = torch.from_numpy(img1).unsqueeze(0)
        '''
        return img, target

    def __len__(self):
        return len(self.data)
  

class KDD_484_H(data.Dataset):

    def __init__(self, root, datatrain,datatest,train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.data = []
        self.targets = []

        #--------------------------------
        data_all = datatrain
        test = datatest
        #--------------------------------
        _, w = data_all.shape
        r2l = None

        print(w)
        if self.train:
            temp_mix = data_all
            arr_1 = temp_mix[:, w-1]
            arr = arr_1.copy()
            arr[arr_1 != 1] = 0

        else:
            #训练集加测试集
            temp_mix = test
            arr_1 = temp_mix[:, w-1]
            arr = arr_1.copy()
            arr[arr_1 != 1] = 0
        #//////////////////////////////////////////////
        #------------------------------------均值
        temp_mean = np.hsplit(data_all,np.array([w-1, w+1]))
        arr_t_mean = temp_mean[1].copy().reshape(-1)
        arr_t_mean[arr_t_mean != 1]=0

        temp_121_mean = np.delete(temp_mean[0], [6], axis=1)

        positive = temp_121_mean[arr_t_mean==1]
        mean_pos = np.mean(positive, axis = 0)

        negtive = temp_121_mean[arr_t_mean == 0]
        mean_neg = np.mean(negtive, axis = 0)
        #----------------------------------------------
        temp = np.hsplit(temp_mix, np.array([w-1, w+1]))

        temp_121 = np.delete(temp[0], [6], axis=1)

        temp_121_1_t = temp_121 - mean_pos
        temp_121_1 = np.abs(temp_121_1_t)
        temp_121_1[temp_121_1 > 0.5] = 1 #差异大于0.5的高亮
        temp_121_1[temp_121_1 <= 0.5] = 0

        temp_121_2_t = temp_121 - mean_neg
        temp_121_2 = np.abs(temp_121_2_t)
        temp_121_2[temp_121_2 > 0.5] = 1
        temp_121_2[temp_121_2 <= 0.5] = 0

        temp_121_3 = temp_121 * temp_121


        temp_t = np.hstack((temp_121, temp_121_1, temp_121_2, temp_121_3))
        temp_data = temp_t[:, :676] 

        self.targets = arr
        self.data = temp_data

        #///////////////////////////////////////////////
        self.data = self.data.reshape(-1, 1, 26, 26)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    train_dataset = KDD_122f(root='../data/')
    