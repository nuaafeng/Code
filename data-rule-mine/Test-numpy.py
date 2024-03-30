import numpy as np
import pandas as pd
import os

filename="data.xlsx"
current_path=os.getcwd()
path=current_path+"/dataset/"+filename
df = pd.read_excel(path)
matrix = df.values
data = np.array(matrix)
data[np.isinf(data)] = np.nan
data[np.isnan(data)] = 0
data_set = np.array(data)
data_set = data_set[0:2000,]
data_set = np.unique(data,axis=0)

print(data_set)