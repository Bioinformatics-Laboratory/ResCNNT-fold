import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置全局随机种子
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算预测概率
        prob = torch.exp(-ce_loss)

        # 计算焦点损失
        focal_loss = self.alpha * (1 - prob) ** self.gamma * ce_loss

        return focal_loss.mean()