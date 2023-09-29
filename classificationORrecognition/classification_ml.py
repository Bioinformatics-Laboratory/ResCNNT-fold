#%%
import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import xgboost as xgb
import torch
import matplotlib.pyplot as plt

from data import get_data_1dim 
from load_nn import get_FoldEmb

def svmmodel(X_train, y_train, X_test, y_test):
    # 设置类别权重（这里使用'balanced'字符串，也可以使用字典形式）
    class_weights = 'balanced'

    # 创建支持向量机分类器
    svm_one_vs_one = OneVsOneClassifier(SVC(kernel='linear', class_weight=None))
    svm_one_vs_all = OneVsRestClassifier(SVC(kernel='linear', class_weight=None))

    # 使用One-vs-One策略进行训练和预测
    svm_one_vs_one.fit(X_train, y_train)
    y_pred_one_vs_one = svm_one_vs_one.predict(X_test)

    # 使用One-vs-All策略进行训练和预测
    svm_one_vs_all.fit(X_train, y_train)
    y_pred_one_vs_all = svm_one_vs_all.predict(X_test)

    # 计算准确率
    accuracy_one_vs_one = accuracy_score(y_test, y_pred_one_vs_one)
    accuracy_one_vs_all = accuracy_score(y_test, y_pred_one_vs_all)

    # 计算每个类别的准确率分布
    # class_accuracies = []
    # for class_label in np.unique(y_test):
    #     class_indices = np.where(y_test == class_label)[0]
    #     class_accuracy = accuracy_score(y_test[class_indices], y_pred_one_vs_all[class_indices])
    #     class_accuracies.append(class_accuracy)

    # print('Class Accuracies:')
    # for class_label, class_accuracy in enumerate(class_accuracies):
    #     print(f'Class {class_label}: {class_accuracy:.2f}')

    #计算f1-score
    f1_one_vs_one = f1_score(y_test, y_pred_one_vs_one, average='weighted')
    f1_one_vs_all = f1_score(y_test, y_pred_one_vs_all, average='weighted')    

    #print("Accuracy of svm_one_vs_one:", accuracy_one_vs_one)
    #print("Accuracy of svm_one_vs_all:", accuracy_one_vs_all)

    return accuracy_one_vs_one, accuracy_one_vs_all, f1_one_vs_one, f1_one_vs_all

def rfmodel(X_train, y_train, X_test, y_test):
    # 设置类别权重（这里使用'balanced'字符串，也可以使用字典形式）
    class_weights = 'balanced'

    # 创建随机森林分类器
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0, class_weight=None)

    # 进行训练
    rf_classifier.fit(X_train, y_train)

    # 进行预测
    y_pred = rf_classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy of Random Forest:", accuracy)
    return accuracy

def lgbmodel(X_train, y_train, X_test, y_test):
    # 定义LightGBM多分类器
    params = {
        'objective': 'multiclass',  # 多分类的目标函数
        'num_class': 38,  # 类别数量
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'verbose': -1
    }

    # 创建lightgbm分类器
    lgb_classifier = lgb.LGBMClassifier(**params)

    # 进行训练
    lgb_classifier.fit(X_train, y_train)

    # 进行预测
    y_pred = lgb_classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy of Random Forest:", accuracy)
    return accuracy    

def xgbmodel(X_train, y_train, X_test, y_test):
    # 定义XGBoost多分类器
    params = {
        'objective': 'multi:softmax',  # 多分类的目标函数
        'num_class': 38,  # 类别数量
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100
    }

    xgb_classifier = xgb.XGBClassifier(**params)

    # 进行训练
    xgb_classifier.fit(X_train, y_train)

    # 进行预测
    y_pred = xgb_classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy of Random Forest:", accuracy)
    return accuracy    

def knnmodel(X_train, y_train, X_test, y_test):
    # 定义KNN多分类器
    n_neighbors = 3  # K值
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # 进行训练
    model.fit(X_train, y_train)

    # 进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy of Random Forest:", accuracy)
    return accuracy    

if __name__ == "__main__":
    start_time=time.time()
    '''
    #使用原始特征
    train_file_dir="data/LE_a_prose_avg.h5"
    test_file_dir="data/LE_b_prose_avg.h5"

    dataset_train, dataset_train_loader = get_data_1dim(train_file_dir,dataset_name='LE',train_test="no")
    dataset_test, dataset_test_loader = get_data_1dim(test_file_dir,dataset_name='LE',train_test="no")

    X_train=dataset_train.tensors[0]
    y_train=dataset_train.tensors[1]
    X_test=dataset_test.tensors[0]
    y_test=dataset_test.tensors[1]

    print("训练集特征张量的shape:", X_train.shape)
    print("训练集标签张量的shape:", y_train.shape)
    print("测试集特征张量的shape:", X_test.shape)
    print("测试集标签张量的shape:", y_test.shape)
    print('num_classes:', torch.max(y_train) + 1)

    '''
    #使用FoldEmb特征
    X_train, y_train, X_test, y_test = get_FoldEmb()
    X_train=X_train.detach().numpy()
    X_test=X_test.detach().numpy()

    #svm
    #print('##############################1-fold##############################')
    accuracy_one_vs_one_1, accuracy_one_vs_all_1, f1_one_vs_one_1, f1_one_vs_all_1=svmmodel(X_train, y_train, X_test, y_test)
    #print('##############################2-fold##############################')
    accuracy_one_vs_one_2, accuracy_one_vs_all_2, f1_one_vs_one_2, f1_one_vs_all_2=svmmodel(X_test, y_test, X_train, y_train)

    accuracy_one_vs_one_avg=(accuracy_one_vs_one_1+accuracy_one_vs_one_2)/2*100
    # f1_one_vs_one_avg=(f1_one_vs_one_1+f1_one_vs_one_2)/2*100
    accuracy_one_vs_all_avg=(accuracy_one_vs_all_1+accuracy_one_vs_all_2)/2*100
    # f1_one_vs_all_avg=(f1_one_vs_all_1+f1_one_vs_all_2)/2*100
    print(f'Accuracy of svm_one_vs_one with 2-fold Cross-Validation: {accuracy_one_vs_one_avg:.2f}%')
    # print(f'f1-score of svm_one_vs_one with 2-fold Cross-Validation: {f1_one_vs_one_avg:.2f}%')
    print(f'*Accuracy of svm_one_vs_all with 2-fold Cross-Validation: {accuracy_one_vs_all_avg:.2f}%')
    # print(f'###f1-score of svm_one_vs_all with 2-fold Cross-Validation: {f1_one_vs_all_avg:.2f}%')
    """
    #rf
    #print('##############################1-fold##############################')
    accuracy_1=rfmodel(X_train, y_train, X_test, y_test)
    #print('##############################2-fold##############################')
    accuracy_2=rfmodel(X_test, y_test, X_train, y_train)

    accuracy_avg=(accuracy_1+accuracy_2)/2*100
    print(f'Accuracy of Random Forest with 2-fold Cross-Validation: {accuracy_avg:.2f}%')

    #lgb
    #print('##############################1-fold##############################')
    accuracy_lgb_1=lgbmodel(X_train, y_train, X_test, y_test)
    #print('##############################2-fold##############################')
    accuracy_lgb_2=lgbmodel(X_test, y_test, X_train, y_train)

    accuracy_lgb_avg=(accuracy_lgb_1+accuracy_lgb_2)/2*100
    print(f'Accuracy of lightgbm with 2-fold Cross-Validation: {accuracy_lgb_avg:.2f}%')

    #xgb
    #print('##############################1-fold##############################')
    accuracy_xgb_1=xgbmodel(X_train, y_train, X_test, y_test)
    #print('##############################2-fold##############################')
    accuracy_xgb_2=xgbmodel(X_test, y_test, X_train, y_train)

    accuracy_xgb_avg=(accuracy_xgb_1+accuracy_xgb_2)/2*100
    print(f'Accuracy of xgboost with 2-fold Cross-Validation: {accuracy_xgb_avg:.2f}%')

    #knn
    #print('##############################1-fold##############################')
    accuracy_knn_1=knnmodel(X_train, y_train, X_test, y_test)
    #print('##############################2-fold##############################')
    accuracy_knn_2=knnmodel(X_test, y_test, X_train, y_train)

    accuracy_knn_avg=(accuracy_knn_1+accuracy_knn_2)/2*100
    print(f'Accuracy of knn with 2-fold Cross-Validation: {accuracy_knn_avg:.2f}%')
    """
    end_time = time.time()
    print('cost %f second' % (end_time - start_time))