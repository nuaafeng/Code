from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
#dataloader
data = np.array(pd.read_csv('covid_train.csv'))
data_test = np.array(pd.read_csv('covid_test.csv'))
shrink = np.array(pd.read_csv('pred-shrink.csv'))[:,[1]]
weight = np.array(pd.read_excel('datatest.xlsx'))[:,[25]]
mean = np.mean(np.array(pd.read_excel('datatest.xlsx'))[:,[27]])
std = np.std(np.array(pd.read_excel('datatest.xlsx'))[:,[27]])
shrink = shrink*std+mean
weight_aft = weight/shrink
x_train = data[:,0:87]
y_train = data[:,[88]]
x_test = data_test[:,0:87]
y_true = np.array(pd.read_csv('pred.csv'))[:,[1]]
print(x_train.shape,y_train.shape)

#PLS
pls1 = PLSRegression(n_components=7)
pls1.fit_transform(x_train,y_train)
y_test = pls1.predict(x_test)
RMSE = mean_squared_error(weight_aft[9000:9999],np.array(pd.read_excel('datatest.xlsx'))[9000:9999,[26]],squared=False)
pca1 = PCA(n_components=2,whiten=True)
X_down = pca1.fit_transform(x_train)
#绘图
fig1 = plt.figure()
epochs = list(range(1, len(weight_aft) + 1))
#plt.plot(epochs, y_true, label = 'Pred',c = 'red')
plt.plot(epochs, weight_aft, label = 'Pred',c = 'red')
plt.plot(epochs,np.array(pd.read_excel('datatest.xlsx'))[:,[26]], label = 'Pred',c = 'blue')
plt.xlabel('Epochs')
plt.ylabel('Pred')
plt.title('pred Over Epochs')
plt.savefig("pred")
print(RMSE)