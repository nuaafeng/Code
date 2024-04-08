import torch
import torch.nn as nn
import torch.functional as F
import math
import pandas as pd
import os
import  torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
d_model = 32#QKV空间维度
n_head = 8#头个数   

#配置文档
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 2332030,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 20000,     # Number of epochs.            
    'batch_size': 256, 
    'learning_rate': 1e-3,              
    'early_stop': 600,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}


#网络结构
class multi_head_attention(nn.Module):

    def __init__(self,d_model,n_head,input_dim):
        super(multi_head_attention,self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_combine = nn.Linear(d_model,d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.LSTM = nn.LSTM(input_dim,5,1)
        self.nn = nn.Linear(5,1)

    def forward(self,q,k,v):
        batch,time,dimension = q.shape
        n_d = self.d_model //self.n_head
        q, k, v = self.w_q(q),self.w_k(k),self.w_v(v)
        q = q.view(batch,time,self.n_head,n_d).permute(0,2,1,3)#更换维度顺序 0，1，2，3 -> 0,2,1,3
        k = k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        v = v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        score = q @ k.transpose(2,3)/math.sqrt(n_d)
        mask = torch.tril(torch.ones(time,time,dtype=bool))
        score = score.masked_fill(mask == 0, float("-inf"))
        score = self.softmax(score) @ v
        score = score.permute(0,2,1,3).contiguous().view(batch, time, dimension)
        attention = self.w_combine(score)
        input = attention @ q
        pred = self.LSTM(input)
        pred = self.nn(pred)
        return pred

#训练器
def trainer(train_loader,valid_loader,model,config,device):
    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.
    optimizer = optim.SGD(net.parameters(),config['learning_rate'])
    writer = SummaryWriter()
    criterion = nn.MSELoss()
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0





a=torch.Tensor(pd.read_excel("/mnt/home/jiangfengrui/Code/MLP/data1.xlsx").values)
b= a[0:1000,:]
b= b.view(4,250,32) #reshape
net = multi_head_attention(d_model,n_head)
output = net(b,b,b)
print(output, output.shape)
attention = multi_head_attention()
out = attention()