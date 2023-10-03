import pandas as pd
import seaborn as sns

train_csv = pd.read_csv("train.csv")
sns.pairplot(train_csv)
