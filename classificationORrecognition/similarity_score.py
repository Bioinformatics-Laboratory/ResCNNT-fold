#%%
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import cityblock
from load_nn import get_FoldEmb

# 设置全局随机种子
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
#%%
#余弦相似性（Cosine Similarity）
def cosine_similarity(x_1, x_2):
    dot_product = np.dot(x_1, x_2)
    norm_x_1 = np.linalg.norm(x_1)
    norm_x_2 = np.linalg.norm(x_2)
    similarity = dot_product / (norm_x_1 * norm_x_2)
    return similarity

#皮尔逊相关系数（Pearson Correlation Coefficient）
def pearson_similarity(x_1, x_2):
    pearson_corr, _ = pearsonr(x_1, x_2)
    return pearson_corr

#曼哈顿相似性（Manhattan Similarity）
def manhattan_similarity(x_1, x_2):
    similarity = 1 / (1 + cityblock(x_1, x_2)) 
    return similarity

#汉明相似性（Hamming Similarity）
def hamming_similarity(x_1, x_2):
    similarity = np.mean([bit1 == bit2 for bit1, bit2 in zip(x_1, x_2)])
    return similarity

#欧几里德相似性（Euclidean Similarity）
def euclidean_similarity(x_1, x_2):
    similarity = 1 / (1 + np.linalg.norm(x_1 - x_2))
    return similarity
#%%
#使用FoldEmb特征
###LE

X_train, y_train, X_test, y_test = get_FoldEmb()

X_train=X_train.detach().numpy()
X_test=X_test.detach().numpy()
X = np.concatenate((X_train, X_test), axis=0)

y_train=y_train.detach().numpy()
y_test=y_test.detach().numpy()
y = np.concatenate((y_train, y_test), axis=0)

###astral_test
"""
X, y = get_FoldEmb()

X = X.detach().numpy()

y = y.detach().numpy()
"""
#%%
Top1_correct=0
Top5_correct=0
for i, item_i in enumerate(X):
    cosine_scores=[]
    for j, item_j in enumerate(X):
        if i==j:
            cosine_scores.append(-999)
        if i!=j:
            cosine_score_single = cosine_similarity(item_i, item_j)
            cosine_scores.append(cosine_score_single)
    # 使用zip函数将两个列表打包在一起，组成元组的列表
    zipped_lists = zip(cosine_scores, y)
    # 对zipped_lists按照第一个元素进行降序排序
    sorted_zipped_lists = sorted(zipped_lists, reverse=True)
    # 将排序后的列表拆解成两个单独的列表
    sorted_cosine_scores, sorted_y = zip(*sorted_zipped_lists)
    
    #print(sorted_cosine_scores)
    #print(len(sorted_cosine_scores))
    #print(sorted_y)
    #print(len(sorted_y))

    if y[i]==sorted_y[0]:
        Top1_correct+=1
    if y[i]==sorted_y[0] or y[i]==sorted_y[1] or y[i]==sorted_y[2]  or y[i]==sorted_y[3] or y[i]==sorted_y[4]:
        Top5_correct+=1
#print(Top1_correct)
#print(Top5_correct)

Top1_accuracy = Top1_correct / len(y) *100
print(f'Top1_accuracy: {Top1_accuracy:.2f}%')

Top5_accuracy = Top5_correct / len(y) *100
print(f'Top5_accuracy: {Top5_accuracy:.2f}%')
# %%
