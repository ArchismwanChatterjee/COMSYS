import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Reading the data sets
train_csv = pd.read_csv("train.csv")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(train_csv.head)

# Preprocessing the data sets
# train_csv['Name'] = train_csv['Name'].astype(float)


def convert(data):
    data = data.copy()
    number = preprocessing.LabelEncoder()
    # data['Name'] = number.fit_transform(data.Name)
    data['Country'] = number.fit_transform(data.Country)
    data = data.fillna(0)
    return data


train_csv = convert(train_csv)

columns_to_exclude = ['id', 'Name', 'Assists']

# Remove the specified columns ('Name' and 'Country') from the DataFrame
train_csv = train_csv.drop(columns=columns_to_exclude)

# Create x (features matrix)
x = train_csv.drop("Value at beginning of 2023/24 season", axis=1)

# Create y (labels)
y = train_csv["Value at beginning of 2023/24 season"]

# median_value1 = x['Value at beginning of 2020/21 season'].mean()
# print(median_value1)

#  Fill the null values in the column with the calculated median
# x['Value at beginning of 2020/21 season'].fillna(median_value1, inplace=True)
# x['Value at beginning of 2020/21 season'].isnull().mean()
# x['Value at beginning of 2020/21 season']
# median_value2 = x['Value at beginning of 2021/22 season'].mean()
# print(median_value2)

#  Fill the null values in the column with the calculated median
# x['Value at beginning of 2021/22 season'].fillna(median_value2, inplace=True)
# x['Value at beginning of 2021/22 season'].isnull().mean()
# x['Value at beginning of 2021/22 season']
# median_value3 = x['Value at beginning of 2022/23 season'].mean()
# print(median_value3)

# # # Fill the null values in the column with the calculated median
# x['Value at beginning of 2022/23 season'].fillna(median_value3, inplace=True)
# x['Value at beginning of 2022/23 season'].isnull().mean()
# x['Value at beginning of 2022/23 season']

nv = x.isnull().mean()
print(nv)

# Choose the right models and hyperparameters
clf = RandomForestRegressor(
    n_estimators=100, min_samples_split=3, max_depth=9, min_samples_leaf=2, random_state=0)


# Keep the default hyperparameters
clf.get_params()

# Fit the model to the training data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.01, random_state=0)
clf.fit(x_train, y_train)


y_preds = clf.predict(x_test)
print(y_preds)

# 11.116914283925817
# 10.881433419785777
# 10.610253840942297
# 10.564188799060044
# 10.495744185210619
# 10.481018627642404
# 10.352546719179935
rmse = np.sqrt(np.mean((y_test-y_preds)**2))
print(f'RMSE: {rmse}')
