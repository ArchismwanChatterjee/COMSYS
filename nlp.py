# reading data
import re
import nltk
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

data = pd.read_csv('train (1).csv', encoding='latin-1')

data.head()

# drop unnecessary columns and rename cols

data.drop(['Unnamed: 0'], axis=1, inplace=True)

data.columns = ['label', 'text']

data.head()
print(data)
# check missing values

data.isna().sum()
# check data shape
data.shape
# check target balance

data['label'].value_counts(normalize=True).plot.bar()
# text preprocessing


# download nltk


nltk.download('all')


# create a list text

text = list(data['text'])


# preprocessing loop


lemmatizer = WordNetLemmatizer()


corpus = []


for i in range(len(text)):

    r = re.sub('[^a-zA-Z]', ' ', text[i])

    r = r.lower()

    r = r.split()

    r = [word for word in r if word not in stopwords.words('english')]

    r = [lemmatizer.lemmatize(word) for word in r]

    r = ' '.join(r)

    corpus.append(r)


# assign corpus to data['text']

data['text'] = corpus

data.head()
print(data)
