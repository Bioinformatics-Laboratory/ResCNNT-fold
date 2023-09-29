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

# 设置全局随机种子
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

def set_activation(activation):
    # Define activation functions
    activation_functions = nn.ModuleDict([
        ["relu", nn.ReLU()], ["lrelu", nn.LeakyReLU()],
        ["sigmoid", nn.Sigmoid()], ["tanh", nn.Tanh()]
    ])
    return activation_functions[activation]

class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512], activation="relu",
                 drop_prob=0, batch_norm=False):
        super().__init__()
        # Define fully-connected layers
        dims = [input_dim] + hidden_dims
        mlp_layers = []
        c=0
        for m, n in zip(dims, dims[1:]):
            c+=1
            #print("{} MLP_layer:\ninput_dim:{}, hidden_dim:{}".format(c,m,n))
            mlp_layers += [nn.Linear(m, n), set_activation(activation)]
            mlp_layers += [nn.BatchNorm1d(n)] if batch_norm else []
            mlp_layers += [nn.Dropout(drop_prob)]

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp(x)

class CNN1D(nn.Module):
    def __init__(self, input_dim=1024, channel_dims=[512, 1024],
                 kernel_sizes=[5, 5], dilations=[1, 1], activation="relu",
                 drop_prob=0.2):
        super().__init__()
        # Define 1D-convolutional layers
        dims = [input_dim] + channel_dims
        #print(dims)
        cnn_layers = []
        for m, n, k, d in zip(dims, dims[1:], kernel_sizes, dilations):
            #print(m,n,k,d)
            cnn_layers += [nn.Sequential(
                nn.Conv1d(m, n, kernel_size=k, padding=int((k-1)/2)*d, dilation=d),
                set_activation(activation),
                nn.BatchNorm1d(n),
                nn.Dropout(drop_prob)
            )]
        self.conv = nn.ModuleList(cnn_layers)

    def forward(self, x):#self, x, seq_mask
        for conv_layer in self.conv:
            x = conv_layer(x)
            #torch.mul(x, seq_mask)x:64*128*256;seq_mask:64*1*256  新张量中的每个元素是两个输入张量对应位置元素的乘积
            #x = torch.mul(x, seq_mask)  # apply mask
        return x

class ResCNN1D(nn.Module):
    def __init__(self, input_dim=1024, channel_dims=[512, 1024, 512, 1024],
                 kernel_sizes=[5, 5], dilations=[1, 1], activation="relu",
                 drop_prob=0.2):
        super().__init__()
        # Define dimensions
        btneck_dim = channel_dims[0]
        conv_dim = channel_dims[1]
        self.apply_initial_conv = False
        if input_dim != conv_dim:
            # Define initial 1D-convolutional layer (upsampling / downsampling)
            self.conv_init = CNN1D(
                input_dim, channel_dims=[conv_dim], kernel_sizes=[1],
                dilations=[1], activation=activation, drop_prob=drop_prob
            )
            self.apply_initial_conv = True
        # Define 1D-convolutional residual blocks with dilation
        residual_blocks = []
        for k, d in zip(kernel_sizes, dilations):
            # Bottleneck residual block (two 1D-convolutions)
            residual_blocks += [CNN1D(
                conv_dim, channel_dims=[btneck_dim, conv_dim],
                kernel_sizes=[k, k], dilations=[d, d], activation=activation,
                drop_prob=drop_prob
            )]
        self.conv_res = nn.ModuleList(residual_blocks)

    def forward(self, x):#self, x, seq_mask
        if self.apply_initial_conv:
            # Compute initial convolution
            x = self.conv_init(x)#x, seq_mask
        # Compute residual blocks
        res = x     # initial residual
        for block in self.conv_res:     # each residual block
            # Compute two convolutional layers
            out = block(res)#res, seq_mask
            # Compute new residual
            res = out + res
        return res

#定义位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=100, dropout=0.1, max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)   

    def forward(self,enc_inputs):
        if torch.cuda.is_available():
            enc_inputs += self.pos_table[:enc_inputs.size(1),:].cuda()
        else:
            enc_inputs += self.pos_table[:enc_inputs.size(1),:]
        return self.dropout(enc_inputs)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):             
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        if torch.cuda.is_available():   
            scores.masked_fill_(attn_mask.bool(), -1e9)
        else:
            scores.masked_fill_(attn_mask, -1e9)                           
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=100, d_k=64, d_v=64, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):  
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) 
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2) 
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)              
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)                                                                      
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) 
        output = self.fc(context)
        if torch.cuda.is_available():
            return nn.LayerNorm(self.d_model).cuda()(output + residual), attn 
        else:                                                   
            return nn.LayerNorm(self.d_model)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=100, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        
    def forward(self, inputs): 
        residual = inputs
        output = self.fc(inputs)
        if torch.cuda.is_available():  
            return nn.LayerNorm(self.d_model).cuda()(output + residual)
        else:
            return nn.LayerNorm(self.d_model)(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model=100, d_ff=2048, d_k=64, d_v=64, n_heads=8):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)                                    
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)                                        

    def forward(self, enc_inputs, enc_self_attn_mask):                                                          
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)                                                                                   
        enc_outputs = self.pos_ffn(enc_outputs)                                       
        return enc_outputs, attn

def get_attn_pad_mask(seq_q, seq_k):                       
    batch_size, len_q ,_= seq_q.size() 
    batch_size, len_k ,_= seq_k.size()
    pad_attn_mask = torch.ones(batch_size,len_q,len_k)    
    return pad_attn_mask  # 扩展成多维度

class Encoder(nn.Module):
    def __init__(self, input_dim=1024, d_model=100, d_ff=2048, d_k=64, d_v=64, n_layers=8, n_heads=8):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(input_dim, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        if torch.cuda.is_available():
            enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs).cuda()
        else:
            enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)              
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

#rescnn+trans
class Transformer(nn.Module):
    def __init__(self, hparams):
        super(Transformer, self).__init__()
        self.rescnn_layers1 = ResCNN1D(
            input_dim=hparams['input_dim'], channel_dims=hparams['channel_dims'],
            kernel_sizes=hparams['kernel_sizes'], dilations=hparams['dilations'], 
            activation=hparams['activation'], drop_prob=hparams['drop_prob']
        )

        self.rescnn_layers2 = ResCNN1D(
            input_dim=hparams['channel_dims'][-1], channel_dims=hparams['channel_dims'],
            kernel_sizes=hparams['kernel_sizes'], dilations=hparams['dilations'], 
            activation=hparams['activation'], drop_prob=hparams['drop_prob']
        )

        self.Encoder1 = Encoder(
            input_dim=hparams['channel_dims'][-1], d_model=hparams['d_model'], 
            d_ff=hparams['d_ff'], d_k=hparams['d_k'], d_v=hparams['d_v'], 
            n_layers=hparams['n_layers'], n_heads=hparams['n_heads']
        )
        '''
        self.rescnn_layers2 = ResCNN1D(
            input_dim=hparams['d_model'], channel_dims=hparams['channel_dims'],
            kernel_sizes=hparams['kernel_sizes'], dilations=hparams['dilations'], 
            activation=hparams['activation'], drop_prob=hparams['drop_prob']
        )

        self.Encoder2 = Encoder(
            input_dim=hparams['channel_dims'][-1], d_model=hparams['d_model'], 
            d_ff=hparams['d_ff'], d_k=hparams['d_k'], d_v=hparams['d_v'], 
            n_layers=hparams['n_layers'], n_heads=hparams['n_heads']
        )
        '''
        self.flatten = nn.Sequential(nn.Flatten())

        # Define fully-connected layers
        self.mlp_layers = MLP(
            input_dim=hparams['d_model'], hidden_dims=hparams['hidden_dims'],
            activation=hparams['activation'], drop_prob=hparams['drop_prob'],
            batch_norm=hparams['batch_norm']
        )

        # Define output layer
        self.out_layer = nn.Linear(int(hparams['hidden_dims'][-1]),
                                   int(hparams['num_classes']))

    def forward(self, x):
        rescnn_inputs1 = x.view(x.size(0), -1, 1) #torch.Size([batch_size, d_model, 1])
        rescnn_outputs1 = self.rescnn_layers1(rescnn_inputs1) #torch.Size([batch_size, channel_dims(最后一层), 1])

        rescnn_outputs2 = self.rescnn_layers2(rescnn_outputs1) #torch.Size([batch_size, channel_dims(最后一层), 1])

        enc_inputs1 = rescnn_outputs2.transpose(1, 2) #torch.Size([batch_size, 1, num_features])
        enc_outputs1, enc_self_attns1 = self.Encoder1(enc_inputs1) #torch.Size([batch_size, 1, d_model])        
        '''
        rescnn_inputs2 = enc_outputs1.transpose(1, 2) #torch.Size([batch_size, d_model, 1])
        rescnn_outputs2 = self.rescnn_layers2(rescnn_inputs2) #torch.Size([batch_size, channel_dims(最后一层), 1])

        enc_inputs2 = rescnn_outputs2.transpose(1, 2) #torch.Size([batch_size, 1, num_features])
        enc_outputs2, enc_self_attns2 = self.Encoder2(enc_inputs2) #torch.Size([batch_size, 1, d_model])  
        '''
        mlp_input=self.flatten(enc_outputs1)

        emb = self.mlp_layers(mlp_input)

        out = self.out_layer(emb)
        return emb, out

