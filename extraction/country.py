import numpy as np
import pandas as pd

con = pd.read_csv('country1_1.csv')
# xtrain = pd.read_csv("train.csv")
# Reading the data sets

xtrain = pd.read_csv("train.csv")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 100)
print(xtrain.head)


def convert(data, con):
    for i in range(25):
        data['Country'] = np.where(
            data['Country'] == con['country'][i], con['frank'][i], data['Country'])
    return data


xtrain = convert(xtrain, con)
print(xtrain)
