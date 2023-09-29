import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import time
import h5py
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# 设置全局随机种子
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

def get_data_1dim(file,dataset_name='LE',train_test="no",batch=64):
    if dataset_name=="LE":
        pattern=re.compile(r'([0-9]_[0-9]+)_[0-9]+_[0-9]+') #LE
    if dataset_name=="LE-superfamily":
        pattern=re.compile(r'([0-9]_[0-9]+_[0-9]+)_[0-9]+') #LE-superfamily 
    if dataset_name=="LE-family":
        pattern=re.compile(r'([0-9]_[0-9]+_[0-9]+_[0-9]+)') #LE-family        
    if dataset_name=="astral":
        pattern=re.compile(r'([a-g]\.[0-9]+).[0-9]+.[0-9]+') #astral

    file = h5py.File(file, 'r') # 打开h5文件

    x=[]
    y=[]
    lens=[]
    names=[]
    count=0
    for key in file.keys():
        count+=1
        #print(count)
        #print(file[key].name)  #获得名称，相当于字典中的key
        #print(file[key][:])  #获得特征值，相当于字典中的value，现在的h5py已经不支持“f[key].value”。
        #print(file[key][:].shape) 

        try:
            #x
            x.append(file[key][:])

            #y
            #pattern=re.compile(r'([0-9]_[0-9]+)_[0-9]+_[0-9]+')#LE
            #pattern=re.compile(r'([a-g]\.[0-9]+).[0-9]+.[0-9]+') #astral
            matchfold=pattern.findall(file[key].name)
            fold=matchfold[0]
            y.append(fold)

            #lens
            length=file[key][:].shape[0]
            lens.append(length)

            #names
            names.append(file[key].name)  
        except:
            key_read = file[key]
            for subkey in key_read.keys():
                #print(key_read[subkey].name)  #获得名称，相当于字典中的key
                #print(key_read[subkey][:])  #获得特征值，相当于字典中的value，现在的h5py已经不支持“f[key].value”。
                #print(key_read[subkey][:].shape) 

                try:
                    #x
                    x.append(key_read[subkey][:])

                    #y
                    #pattern=re.compile(r'([0-9]_[0-9]+)_[0-9]+_[0-9]+')#LE
                    #pattern=re.compile(r'([a-g]\.[0-9]+).[0-9]+.[0-9]+') #astral
                    matchfold=pattern.findall(key_read[subkey].name)
                    fold=matchfold[0]
                    y.append(fold)

                    #lens
                    length=key_read[subkey][:].shape[0]
                    lens.append(length)

                    #names
                    names.append(key_read[subkey].name)
                except:
                    subkey_read = key_read[subkey]
                    for subsubkey in subkey_read.keys():
                        #print(subkey_read[subsubkey].name)  #获得名称，相当于字典中的key
                        #print(subkey_read[subsubkey][:])  #获得特征值，相当于字典中的value，现在的h5py已经不支持“f[key].value”。
                        #print(subkey_read[subsubkey][:].shape) 

                        try:
                            #x
                            x.append(subkey_read[subsubkey][:])

                            #y
                            #pattern=re.compile(r'([0-9]_[0-9]+)_[0-9]+_[0-9]+')#LE
                            #pattern=re.compile(r'([a-g]\.[0-9]+).[0-9]+.[0-9]+') #astral
                            matchfold=pattern.findall(subkey_read[subsubkey].name)
                            fold=matchfold[0]
                            y.append(fold)

                            #lens
                            length=subkey_read[subsubkey][:].shape[0]
                            lens.append(length)

                            #names
                            names.append(subkey_read[subsubkey].name)
                        except:
                            subsubkey_read = subkey_read[subsubkey]
                            for subsubsubkey in subsubkey_read.keys():
                                #print(subsubkey_read[subsubsubkey].name)  #获得名称，相当于字典中的key
                                #print(subsubkey_read[subsubsubkey][:])  #获得特征值，相当于字典中的value，现在的h5py已经不支持“f[key].value”。
                                #print(subsubkey_read[subsubsubkey][:].shape) 
                                try:
                                    #x
                                    x.append(subsubkey_read[subsubsubkey][:])

                                    #y
                                    #pattern=re.compile(r'([0-9]_[0-9]+)_[0-9]+_[0-9]+')#LE
                                    #pattern=re.compile(r'([a-g]\.[0-9]+).[0-9]+.[0-9]+') #astral
                                    matchfold=pattern.findall(subsubkey_read[subsubsubkey].name)
                                    fold=matchfold[0]
                                    y.append(fold)

                                    #lens
                                    length=subsubkey_read[subsubsubkey][:].shape[0]
                                    lens.append(length)

                                    #names
                                    names.append(subsubkey_read[subsubsubkey].name)
                                except:
                                    subsubsubkey_read = subsubkey_read[subsubsubkey]
                                    for subsubsubsubkey in subsubsubkey_read.keys():
                                        #print(subsubsubkey_read[subsubsubsubkey].name)  #获得名称，相当于字典中的key
                                        #print(subsubsubkey_read[subsubsubsubkey][:])  #获得特征值，相当于字典中的value，现在的h5py已经不支持“f[key].value”。
                                        #print(subsubsubkey_read[subsubsubsubkey][:].shape) 

                                        #x
                                        x.append(subsubsubkey_read[subsubsubsubkey][:])

                                        #y
                                        #pattern=re.compile(r'([0-9]_[0-9]+)_[0-9]+_[0-9]+')#LE
                                        #pattern=re.compile(r'([a-g]\.[0-9]+).[0-9]+.[0-9]+') #astral
                                        matchfold=pattern.findall(subsubsubkey_read[subsubsubsubkey].name)
                                        fold=matchfold[0]
                                        y.append(fold)

                                        #lens
                                        length=subsubsubkey_read[subsubsubsubkey][:].shape[0]
                                        lens.append(length)

                                        #names
                                        names.append(subsubsubkey_read[subsubsubsubkey].name)

        #print(np.array(x).shape)

    y_fold=y #原始标签

    y=preprocessing.LabelEncoder().fit_transform(y) #标签标准化

    # 打印标签编码后的结果
    # label_dict=dict(zip(y, y_fold))
    # name_dict=dict(zip(y, names))
    # print(label_dict)
    # print(name_dict)

    x=torch.from_numpy(np.array(x))
    y=torch.from_numpy(np.array(y))
    seq_len=torch.from_numpy(np.array(lens))
    name=np.array(names)
    #print(type(x),x.dtype,x.shape,sep=',')#<class 'torch.Tensor'>,torch.float32,torch.Size([321, 490, 6165])
    #print(type(y),y.dtype,y.shape,sep=',')#<class 'torch.Tensor'>,torch.int64,torch.Size([321])
    #print(type(seq_len),seq_len.dtype,seq_len.shape,sep=',')#<class 'torch.Tensor'>,torch.int64,torch.Size([321])
    #print(type(name),name.dtype,name.shape,sep=',')#<class 'numpy.ndarray'>,<U28,(321,)

    if train_test=="yes":
        #随机分出训练集和测试集
        x_tra, x_val, y_tra, y_val = train_test_split(x, y, test_size=0.1, random_state=0)

        dataset_train=TensorDataset(x_tra,y_tra) 
        dataset_test=TensorDataset(x_val,y_val) 

        # DataLoader进行数据封装
        dataset_train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=True)
        dataset_test_loader = DataLoader(dataset_test, batch_size=batch, shuffle=True)

        return dataset_train, dataset_test, dataset_train_loader, dataset_test_loader
    if train_test=="no":
        dataset=TensorDataset(x,y) 

        dataset_loader = DataLoader(
            dataset, 
            batch_size=batch, 
            shuffle=True,
            #drop_last=True, 
            #num_workers=0, # 进程数, 0表示只有主进程
            #collate_fn=collate_dataset #
        )

        return dataset, dataset_loader 

def main(choose='astral_1dim',batch=64):
    #=======LE=======#
    if choose=='LE_1dim':
        train_file_dir="embedding/ProSE/LE/321a_avg.h5"
        test_file_dir="embedding/ProSE/LE/321b_avg.h5"

        dataset_train, dataset_train_loader = \
            get_data_1dim(train_file_dir,dataset_name='LE',train_test="no",batch=batch)
        dataset_test, dataset_test_loader = \
            get_data_1dim(test_file_dir,dataset_name='LE',train_test="no",batch=batch) 
        
        X_train=dataset_train.tensors[0]
        y_train=dataset_train.tensors[1]
        X_test=dataset_test.tensors[0]
        y_test=dataset_test.tensors[1]
        
        # print("训练集特征张量的shape:", X_train.shape)
        # print("训练集标签张量的shape:", y_train.shape)
        # print("测试集特征张量的shape:", X_test.shape)
        # print("测试集标签张量的shape:", y_test.shape)
        # print('num_classes:', torch.max(y_train) + 1)

        return X_train, y_train, X_test, y_test, dataset_train_loader, dataset_test_loader

    #=======LE=======#
    if choose=='LE-superfamily_1dim':
        train_file_dir="embedding/ProSE/LE/434a_avg.h5"
        test_file_dir="embedding/ProSE/LE/434b_avg.h5"

        dataset_train, dataset_train_loader = \
            get_data_1dim(train_file_dir,dataset_name='LE-superfamily',train_test="no",batch=batch)
        dataset_test, dataset_test_loader = \
            get_data_1dim(test_file_dir,dataset_name='LE-superfamily',train_test="no",batch=batch) 
        
        X_train=dataset_train.tensors[0]
        y_train=dataset_train.tensors[1]
        X_test=dataset_test.tensors[0]
        y_test=dataset_test.tensors[1]
        
        # print("训练集特征张量的shape:", X_train.shape)
        # print("训练集标签张量的shape:", y_train.shape)
        # print("测试集特征张量的shape:", X_test.shape)
        # print("测试集标签张量的shape:", y_test.shape)
        # print('num_classes:', torch.max(y_train) + 1)

        return X_train, y_train, X_test, y_test, dataset_train_loader, dataset_test_loader

    #=======LE=======#
    if choose=='LE-family_1dim':
        train_file_dir="embedding/ProSE/LE/555a_avg.h5"
        test_file_dir="embedding/ProSE/LE/555b_avg.h5"

        dataset_train, dataset_train_loader = \
            get_data_1dim(train_file_dir,dataset_name='LE-family',train_test="no",batch=batch)
        dataset_test, dataset_test_loader = \
            get_data_1dim(test_file_dir,dataset_name='LE-family',train_test="no",batch=batch) 
        
        X_train=dataset_train.tensors[0]
        y_train=dataset_train.tensors[1]
        X_test=dataset_test.tensors[0]
        y_test=dataset_test.tensors[1]
        
        # print("训练集特征张量的shape:", X_train.shape)
        # print("训练集标签张量的shape:", y_train.shape)
        # print("测试集特征张量的shape:", X_test.shape)
        # print("测试集标签张量的shape:", y_test.shape)
        # print('num_classes:', torch.max(y_train) + 1)

        return X_train, y_train, X_test, y_test, dataset_train_loader, dataset_test_loader

    #=======astral=======#
    if choose=='astral_1dim':
        train_file_dir="embedding/ProSE/astral_95_cdhit2_avg.h5"

        dataset_train, dataset_test, dataset_train_loader, dataset_test_loader = \
            get_data_1dim(train_file_dir,dataset_name='astral',train_test="yes",batch=batch)
        
        X_train=dataset_train.tensors[0]
        y_train=dataset_train.tensors[1]
        X_test=dataset_test.tensors[0]
        y_test=dataset_test.tensors[1]
        
        # print("训练集特征张量的shape:", X_train.shape)
        # print("训练集标签张量的shape:", y_train.shape)
        # print("测试集特征张量的shape:", X_test.shape)
        # print("测试集标签张量的shape:", y_test.shape)
        # print('num_classes:', torch.max(y_train) + 1)

        return X_train, y_train, X_test, y_test, dataset_train_loader, dataset_test_loader
