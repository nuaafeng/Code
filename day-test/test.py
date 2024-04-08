import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc4 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc5 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc6 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc7 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc8 = nn.Linear(hidden_size, output_size).to(device)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 2
hidden_size = 128
output_size = 4
data = torch.tensor(np.load(r"data/points_results.npy")).float().to(device)
data = data[~torch.isnan(data).any(1)]
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.apply(weight_init)
#torch中的apply函数通过可以不断遍历model的各个模块，并将weight_init函数应用在这些Module上

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.9)

# 假设你已经有一个名为data的数据集，包含1000000个样本
# data应该是一个Tensor，每行是一个样本，前两列是输入，后四列是输出
# 使用DataLoader来加载数据集
# DataLoader的batch_size可以根据你的实际情况进行调整
input = data[:,]
data_loader = DataLoader(TensorDataset(data[:, :2], data[:, 3:]), batch_size=32, shuffle=True)

# 训练模型
import os
if not os.path.isdir('./models'):
    os.mkdir('./models')
num_epochs = 10000
best_loss = 1
writer = SummaryWriter()

for epoch in range(num_epochs):
    model.train()
    loss_record =[]
    train_pbar = tqdm(data_loader,position=0,leave=True)
    for x, y in train_pbar:
        # 前向传播
        # Q1 output 不是单位向量
        # Q2 1e6个点太多了，所以nan
        outputs = model(x)
        loss = criterion(outputs, y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss.detach().item())
        train_pbar.set_description(f'Epoch [{epoch+1}/{epoch}]')
        train_pbar.set_postfix({'loss': loss.detach().item()})
    mean_train_loss = sum(loss_record)/len(loss_record)
    writer.add_scalar('Loss/train',mean_train_loss,epoch)
    
    if loss<=best_loss:
        best_loss = loss
        i = 0
    else:
        i = i+1
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if i >= 100:
        print('STOP train')
        break
writer.close
# 保存模型
torch.save(model.state_dict(), 'dune_model/model.pth')
