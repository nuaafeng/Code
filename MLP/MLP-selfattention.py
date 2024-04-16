import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import math
from sklearn.preprocessing import StandardScaler
#数据集定义
class Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
#utility function

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)# 对模型的所有线性层应用 Xavier 初始化方法

def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader,model,device):
    net.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds,dim=0).numpy()       
    return preds

#特征选择
def select_feat(train_data, valid_data, test_data, select_all=True):
    y_train, y_valid = train_data[:,26], valid_data[:,26]#-1表示倒数第一列
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))#特征个数
    else:
        feat_idx = [8,9,10,11,12,13,14,17,25,28] # TODO: Select suitable feature columns.
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid


#网络结构
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,input_dim),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=10,num_heads=1)
        self.out = nn.Linear(10,1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.model(x)
        x,x_weihts= self.attention(x,x,x)
        x = self.relu(self.out(x))
        x = x.squeeze(1)
        return x

#训练器
def trainer(train_loader,valid_loader,net,config,device):
    #创建模型文件夹
    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.
    #优化器
    optimizer = optim.SGD(net.parameters(),config['learning_rate'])
    writer = SummaryWriter()
    #误差函数
    criterion = nn.MSELoss()
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    #训练回路
    for epoch in range(n_epochs):
        net.train()
        loss_record = []
        train_pbar = tqdm(train_loader,position=0,leave=True)
        for x,y in train_pbar:
            optimizer.zero_grad() #梯度清零
            x,y = x.to(device),y.to(device)
            pred = net(x)
            loss = criterion(pred,y)
            loss.backward() #梯度
            optimizer.step() #更新参数
            step += 1
            loss_record.append(loss.detach().item())
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train',mean_train_loss,step)
        #测试
        net.eval()
        all_loss = []
        for x,y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():#不跟踪梯度节省内存
                pred = net(x)
                loss = criterion(pred, y)
            all_loss.append(loss.item())
        mean_valid_loss = sum(all_loss)/len(all_loss)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(net.state_dict(),config['save_path'])
            print('model saved')
            early_stop_count = 0
        else:
            early_stop_count +=1
        
        if early_stop_count >= config['early_stop']:
            print('model is not improving,STOP!')
            return

#配置文档
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 102110,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 20000,     # Number of epochs.            
    'batch_size': 256, 
    'learning_rate': 1e-3,              
    'early_stop': 600,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model-selfattention.ckpt'  # Your model will be saved here.
}

#数据加载
same_seed(config['seed'])
train_data, test_data = pd.read_excel('./data1.xlsx').values, pd.read_excel('./datatest.xlsx').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
#标准化
scaler = StandardScaler() 
train_data, test_data = scaler.fit_transform(train_data), scaler.fit_transform(test_data)
train_data, valid_data = scaler.fit_transform(train_data),scaler.fit_transform(valid_data)
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")
#DATASET生成
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])
print(f'number of features: {x_train.shape[1]}')
train_dataset, valid_dataset, test_dataset = Dataset(x_train, y_train), \
                                            Dataset(x_valid, y_valid), \
                                            Dataset(x_test)

#转化为pytorch格式
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)


net = My_Model(input_dim=x_train.shape[1]).to(device)
net.apply(xavier_init)
trainer(train_loader, valid_loader, net, config, device)

#测试
import csv
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred_attention.csv')