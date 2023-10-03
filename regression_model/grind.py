from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def fill(data: pd.DataFrame):
    attribute = ['Value at beginning of 2020/21 season',
                 'Value at beginning of 2021/22 season',
                 'Value at beginning of 2022/23 season']

    for i in attribute:
        filler = data[attribute].mean()
        data[attribute] = data[attribute].fillna(value=filler)
    return data


def fillCountryValue(countries: pd.DataFrame, data: pd.DataFrame):
    for i in range(25):
        data['Country'] = np.where(
            data['Country'] == countries['country'][i], countries['frank'][i], data['Country'])
    data['Country'] = np.where(pd.to_numeric(
        data['Country'], errors='coerce').isna(), 0, data['Country'])
    return data


def handleNegs(preds):
    # find mean of pos values
    df = pd.DataFrame(preds)
    filler = df.loc[df[0] > 0].quantile(0.094)

    # fit mean to neg values
    for i in range(len(preds)):
        if preds[i] <= 0:
            preds[i] = filler
    return preds


data = pd.read_csv("train.csv")
data = fill(data)
countries = pd.read_csv("country1_6.csv")
data = fillCountryValue(countries, data)
test = pd.read_csv("test.csv")
test = fill(test)
toDrop = ['id', 'Expected Goal Contributions',
          'Blocks', 'Clearances', 'Interceptions']
worth = 'Value at beginning of 2023/24 season'
name = 'Name'

train = data.iloc[:1035]
test = data.iloc[1035:]

train_data = train.drop(columns=toDrop+list([worth, name]))
train_ans = train[worth]
test_data = test.drop(columns=toDrop+list([worth, name]))
test_ans = test[worth]

min = 100
min_i = 0
for i in range(1, 2):
    model = LinearRegression()
    model.fit(train_data, train_ans)
    preds = model.predict(test_data)
    preds = handleNegs(preds)
    rmse = np.sqrt(np.mean((test_ans-preds)**2))

    print(f'RMSE:{i}     {rmse}\n')
    if rmse < min:
        min = rmse
        min_i = i

print(f'{min_i} : {min}')
# Ridge : alpha=0, fit_intercept=False, tol=0.00001, solver='auto' : 8.392547340420043
# Lasso : alpha=0, fit_intercept=False, selection='random' : 8.399944693810806
# 7.716582908442377
