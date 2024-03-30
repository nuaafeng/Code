import PCA as pc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#FCM调用函数
def FCM(X, c_clusters=3, m=2, eps=10):
    membership_mat = np.random.random((len(X), c_clusters))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])

    while True:
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, X), np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])
        
        n_c_distance_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x-c, 2)
        
        new_membership_mat = np.zeros((len(X), c_clusters))
        
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** (2 / (m - 1)))
        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        membership_mat =  new_membership_mat
    return np.argmax(new_membership_mat, axis=1)

#FCM结果评估函数
def evaluate(y, t):
    a, b, c, d = [0 for i in range(4)]
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] == y[j] and t[i] == t[j]:
                a += 1
            elif y[i] == y[j] and t[i] != t[j]:
                b += 1
            elif y[i] != y[j] and t[i] == t[j]:
                c += 1
            elif y[i] != y[j] and t[i] != t[j]:
                d += 1
    return a, b, c, d

def external_index(a, b, c, d, m):
    JC = a / (a + b + c)
    FMI = np.sqrt(a**2 / ((a + b) * (a + c)))
    RI = 2 * ( a + d ) / ( m * (m + 1) )
    return JC, FMI, RI

def evaluate_it(y, t):
    a, b, c, d = evaluate(y, t)
    return external_index(a, b, c, d, len(y))

if __name__ == "__main__":
#数据处理
    df = pd.read_excel("data.xlsx")
    df1 = pd.read_excel("data1.xlsx")
    matrix = df.values
    data = np.array(matrix)
    data[np.isinf(data)] = np.nan
    data[np.isnan(data)] = 0
    data = data[0:2000]
    data1 = np.array(df.values)
    data1[np.isinf(data1)] = np.nan
    data1[np.isnan(data1)] = 0
    data = np.array(data)
    #data = np.unique(data,axis=0)#去除data中重复的项
    data = data[:,[0,1,2,3,8,9,10,11,12,13,14,15,16,17]]
    data1 = np.array(data1)
    train_FCM = data1[:,[0,1,2,3,8,9,10,11,12,13,14,15,16,17]]
    #train_FCM = np.unique(train_FCM,axis=0)
    np.random.shuffle(data)
    train = data
    Test1 = data1[0:400,[0,1,2,3,8,9,10,11,12,13,14,15,16,17]]
    #Test1 = np.tile(Test1,(5,1)) 重复5次前400行
    Test2 = data1[1600:2000,[0,1,2,3,8,9,10,11,12,13,14,15,16,17]]
    Test3 = data1[0:400,[0,1,2,3,8,9,10,11,12,13,14,15,16,17]]

#PCA降维
    pca1 = PCA(n_components=2,whiten=True)
    res = pca1.fit_transform(data)
    ratio = pca1.explained_variance_ratio_
    print("各特征的权重为: ratio = ",ratio)
    print("使用sklearn.decomposition.PCA 验证的结果为: res = ", res)
    X_new = pca1.transform(train)
    X_1 = pca1.transform(Test1)
    X_2 = pca1.transform(Test2)
    X_3 = pca1.transform(Test3)
    fig = plt.figure(0)
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(X_1[:,0],X_1[:,1],X_1[:,2],marker='^')
    #ax.scatter(X_new[:,0],X_new[:,1],X_new[:,2],marker='o')
    #ax.scatter(X_2[:,0],X_2[:,1],X_2[:,2],marker='*')
    plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
    plt.scatter(X_1[:, 0], X_1[:, 1],marker='^')
    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    fig1 = plt.figure(1)
    plt.scatter(X_1[:, 0], X_1[:, 1],marker='^')
    plt.scatter(X_2[:, 0], X_2[:, 1],marker='*')
    plt.scatter(X_3[:, 0], X_3[:, 1],marker='o')
    plt.savefig("figs/picture.png",dpi=300)
#FCM聚类
test_y = FCM(train_FCM)
pca1 = PCA(n_components=2,whiten=True)
X_reduced = pca1.fit_transform(train_FCM)
fig2 = plt.figure(2)
""" plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=test_y, cmap=plt.cm.Set1) """
plt.scatter(X_1[:, 0], X_1[:, 1],marker='^')
plt.scatter(X_2[:, 0], X_2[:, 1],marker='*')
plt.savefig("figs/picture1.png",dpi=300)
model = KMeans(n_clusters=3,n_init=10) #构造聚类器
model.fit(X_reduced) #拟合聚类模型
label_pred = model.labels_
x0 = X_reduced[label_pred == 0]
x1 = X_reduced[label_pred == 1]
x2 = X_reduced[label_pred == 2]
fig3 = plt.figure(3)
plt.scatter(x0[:, 0], x0[:, 1], c='red')
plt.scatter(x1[:, 0], x1[:, 1], c='blue')
plt.scatter(x2[:, 0], x2[:, 1], c='green')
plt.savefig("figs/picture2.png",dpi=300)
plt.show()