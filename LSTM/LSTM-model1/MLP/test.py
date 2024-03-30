import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms

#读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)#feature 1000,2
    indices = list(range(num_examples))
    #random.shuffle(indices)#打乱顺序
    for i in range(0, num_examples, batch_size):
        batch_indeices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        yield features[batch_indeices], labels[batch_indeices]

df = pd.read_excel("data1.xlsx")
data = np.array(df.values)
X_train = data[0:1500,0:17]
Y_train = data[0:1500,[18]]
X = data[0:1500,0:17]
Y_true = data[0:1500,[18]]
X_train2 = torch.tensor(data[0:1500,[8,9,10,11,12,13,14,17,24,25]])
X_train2 = X_train2.double()
Y_train2 = torch.tensor(data[0:1500,26])
Y_train2 = Y_train2.double()
X_train2 = torch.tensor(data[0:1500,[8,9,10,11,12,13,14,17,24,25]],dtype=torch.float32)
Y_train2 = torch.tensor(data[0:1500,26],dtype=torch.float32)
X2 = data[0:2000,[8,9,10,11,12,13,14,17,24,25]]
Y_true2 = data[0:2000,26]
Y_true2 = torch.tensor(data[0:1500,26],dtype=torch.float32)
mean = torch.mean(Y_true2, dim=0)
std = torch.std(Y_true2, dim=0)
Y_true2 = F.normalize(Y_true2,dim=0)
Y_TEST = (Y_true2 * std) + mean
X_train2 = F.normalize(X_train2,dim=0)


epochs = list(range(1, len(Y_true2) + 1))
plt.plot(epochs, Y_TEST, label = 'Pred',c = 'red')
plt.plot(epochs, Y_train2, label = 'test',c = 'green')
plt.xlabel('Epochs')
plt.ylabel('Pred')
plt.title('pred Over Epochs')
plt.show()
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10,10),
            nn.ReLU(inplace=True),
            nn.Linear(10,5),
            nn.ReLU(inplace=True),
            nn.Linear(5,1)
        )
    def forward(self,x):
        x = self.model(x)
        return x

model = MLP()
model.load_state_dict(torch.load('./model.ckpt'))
# 打印模型的权重系数
print("模型的权重系数：")
for name, param in model.named_parameters():
    print(f"{name}: {param}")
    if 'bias' in name:
        print(f"Layer: {name}, Size: {param.size()}, Values: {param}")





