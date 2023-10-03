
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
import csv


def fill(data: pd.DataFrame):
    attribute = ['Value at beginning of 2020/21 season',
                 'Value at beginning of 2021/22 season',
                 'Value at beginning of 2022/23 season']

    for i in attribute:
        filler = 0
        data[attribute] = data[attribute].fillna(value=filler)
    return data


def fillCountryValue(countries: pd.DataFrame, data: pd.DataFrame):
    for i in range(25):
        data['Country'] = np.where(
            data['Country'] == countries['country'][i], countries['frank'][i], data['Country'])
    data['Country'] = np.where(pd.to_numeric(
        data['Country'], errors='coerce').isna(), 0, data['Country'])
    return data


data = pd.read_csv("train.csv")
fill(data)
test = pd.read_csv("test.csv")
fill(test)
countries = pd.read_csv("country1_1.csv")
data = fillCountryValue(countries, data)
test = fillCountryValue(countries, test)
toDrop = ['id']
worth = 'Value at beginning of 2023/24 season'
name = 'Name'

train_data = data.drop(columns=toDrop+list([worth, name]))
train_ans = data[worth]
test_data = test.drop(columns=toDrop)


model = LinearRegression()
model.fit(train_data, train_ans)

preds = model.predict(test_data)

for i in range(len(preds)):
    preds[i] = abs(preds[i])

final = pd.read_csv("sample_submission.csv")
final['label'] = preds

df = pd.DataFrame(final)
print(df)
df.to_csv("submission.csv", index=False)
