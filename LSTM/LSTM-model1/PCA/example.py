import PCA as pc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_excel("data.xlsx")

matrix = df.values

data = np.array(matrix)
data[np.isinf(data)] = np.nan
data[np.isnan(data)] = 0

if __name__ == "__main__":
    data = np.array(data)

#    pca = pc.PCA(n_components=’mle‘)
#   pca.fit(data,rowvar=False)
#   res = pca.transform(data,rowvar=False)
#    ratio = pca.variance_ratio(only=True)
#    print("各特征的权重为: ratio = ",ratio)
#    print("使用本库进行计算得到的PCA降维结果为: res = ", res)

    pca1 = PCA(n_components=0.95)
    res = pca1.fit_transform(data)
    ratio = pca1.explained_variance_ratio_
    print("各特征的权重为: ratio = ",ratio)
    print("使用sklearn.decomposition.PCA 验证的结果为: res = ", res)
    X_new = pca1.transform(data)
    fig = plt.figure()
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
    plt.show()
    

