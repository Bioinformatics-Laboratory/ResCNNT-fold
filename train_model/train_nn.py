import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import h5py
import numpy as np
from sklearn.utils import class_weight
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from data import main
from model import Transformer
from loss import FocalLoss

# 设置全局随机种子
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# 训练测试模型
def train_2dim(X_train, y_train, X_test, y_test, dataset_train_loader, dataset_test_loader, save_model_dir):
    # 使用交叉熵损失函数
    #class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train.numpy())
    #class_weights=torch.tensor(class_weights,dtype=torch.float)
    #criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(alpha=0.25, gamma=5.0)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        #criterion = criterion.to(device)
    # 定义优化器
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    # 训练模型
    loss_list=[]  #为了后续画出损失图
    best_loss=10000
    best_epoch=0

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss=0.0
        for inputs, labels in dataset_train_loader:
            optimizer.zero_grad() # 清空上一次梯度
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            emb, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # 误差反向传递 只需要调用.backward()即可
            optimizer.step() # 优化器参数更新

            running_loss+=loss.item()
        loss_list.append(running_loss)
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(running_loss))

        if running_loss<best_loss:
            best_loss=running_loss
            best_epoch=epoch
    print('best_loss::|',best_loss,'---best_epoch::|',best_epoch)
    print(loss_list)
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            X_test = X_test.cuda()
            y_test = y_test.cuda()
        emb, outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total * 100
        print(f'Test Accuracy: {accuracy:.2f}%')

    #保存模型
    torch.save(model, save_model_dir)

    return accuracy


if __name__ == "__main__":
    start_time=time.time()


    ### 提取数据
    Data = main(choose='astral_1dim',batch=512)
    X_train, y_train, X_test, y_test, dataset_train_loader, dataset_test_loader = Data
    
    '''
    ### 解决数据不平衡问题(采样)
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    X_train=torch.from_numpy(np.array(X_train))
    y_train=torch.from_numpy(np.array(y_train))
    '''

    ### 定义模型
    
    # 定义FCNN超参数字典
    '''
    hparams = {
        'input_dim': X_train.shape[1],  # 输入特征数
        'hidden_dims': [2048,1024],  # 隐藏层的大小
        'num_classes': torch.max(y_train) + 1,  # 类别数
        'activation': "relu",
        'drop_prob': 0.5,
        'batch_norm': True,
    }
    '''
    # 定义CNN超参数字典
    '''
    hparams = {
        'seq_len': 1,
        'input_dim': X_train.shape[1],  # 输入特征数
        'channel_dims': [1024,512],
        'kernel_sizes': [5, 5],
        'dilations': [1, 1],
        'hidden_dims': [2048,1024],  # 隐藏层的大小
        'num_classes': torch.max(y_train) + 1,  # 类别数
        'activation': ["relu","relu"],
        'drop_prob': 0.2,
        'batch_norm': True,
    }
    '''
    # 定义CNN2超参数字典
    '''
    hparams = {
        'seq_len': 1,
        'input_dim': X_train.shape[1],  # 输入特征数
        'channel_dims': [1024],
        'kernel_sizes': [5, 5],
        'dilations': [1, 1],
        'hidden_dims': [1024],  # 隐藏层的大小
        'num_classes': torch.max(y_train) + 1,  # 类别数
        'activation': ["relu","relu"],
        'drop_prob': 0.2,
        'batch_norm': True,
    }
    '''
    #transformer
    hparams = {
        'input_dim': X_train.shape[1],  # 输入特征数

        'd_model': 1024, # 字 Embedding 的维度
        'd_ff': 2048, # 前向传播隐藏层维度
        'd_k': 64, # K(=Q)的维度
        'd_v': 64, # V的维度 
        'n_layers': 4, # 有多少个encoder和decoder
        'n_heads': 4, # Multi-Head Attention

        'channel_dims': [1024, 2048],
        'kernel_sizes': [5, 5],
        'dilations': [1, 1],        

        'hidden_dims': [2048,1024],  # 隐藏层的大小
        'num_classes': torch.max(y_train) + 1,  # 类别数
        'activation': "relu",
        'drop_prob': 0.2,
        'batch_norm': True,
    }
    
    # 输出模型
    #model=MultiClassMLP(hparams)
    #model=MultiClassCNN(hparams)
    #model=MultiClassVGG()
    model=Transformer(hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    print(model)

    ### 训练模型
    save_model_dir='data/model_Focalloss_batch512_epoch50.pkl'
    accuracy=train_2dim(X_train, y_train, X_test, y_test, dataset_train_loader, dataset_test_loader, save_model_dir)
    
    #with open('output.txt', 'a+') as f:
        #f.write(str(accuracy)+"\n")


    end_time = time.time()
    print('cost %f second' % (end_time - start_time))
