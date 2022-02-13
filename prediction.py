from nltk.util import pr
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GRU
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import re

from tensorflow.python.keras.backend import dtype

def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub(' ',line)
    return line
def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]

    return " ".join(text)

dtypes = {'title': 'str', 'text':'str','category': 'int'}
df = pd.read_csv('G:\\Polytech project\\project\\predict_example.csv',header=0,names=['title','text'],delimiter=";",dtype=dtypes)
# Установите 50 000 наиболее часто используемых слов
MAX_NB_WORDS = 50000
 # Максимальная длина каждого cut_review
MAX_SEQUENCE_LENGTH = 1000
 # Установите размеры слоя Embeddingceng
EMBEDDING_DIM = 450
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
#file = open("G:\\Polytech project\\project\\predict_example.txt", "r", encoding='utf-8')
#txt = file.read()
model = load_model('G:\\Polytech project\\project\\best_model_lstm.h5')
result = model.predict(X)
cat_id= result.argmax(axis=1)[0]

print("Категория текста = " + str(cat_id))