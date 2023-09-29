import numpy as np
import torch

from data import main
from model import Transformer

# 设置全局随机种子
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

def get_FoldEmb():
    model = torch.load('model/model.pkl', map_location=lambda storage, loc: storage)
    model = model.to('cpu')
    print(model)

    X_train, y_train, X_test, y_test, dataset_train_loader, dataset_test_loader = main(choose="LE_1dim",batch=512)

    LE_a_FoldEmb, LE_a_outputs = model(X_train)
    LE_b_FoldEmb, LE_b_outputs = model(X_test)
    #print(LE_a_FoldEmb)
    #print(LE_a_FoldEmb.shape)
    #print(LE_b_FoldEmb)
    #print(LE_b_FoldEmb.shape)

    X_train=LE_a_FoldEmb
    y_train=y_train
    X_test=LE_b_FoldEmb
    y_test=y_test

    print("训练集特征张量的shape:", X_train.shape)
    print("训练集标签张量的shape:", y_train.shape)
    print("测试集特征张量的shape:", X_test.shape)
    print("测试集标签张量的shape:", y_test.shape)
    print('num_classes:', torch.max(y_train) + 1)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    get_FoldEmb()