from typing import Text
import pandas as pd
from tqdm import tqdm
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from  sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import razdel
import re
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

def remove_punctuation(line):
    line = re.sub(r"https?://[^,\s]+,?", "", line) #Удаляем ссылки
    #line = re.sub(r"(\s|\b).*?(\.ru|\.com|\.org|\.net).*?\s", "", line) #Удаляем не HTTP ссылки
    line = re.sub(r'[^\w\s]','', line) #Удаляем пунктуацию
    line = re.sub(r"\d+", "", line, flags=re.UNICODE) #Удаляем цифры
    line = re.sub("^\s+|\n|\r|\s+$", '', line) # Удаляем несколько пробелов и перенос строки
    return line
    
def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

path = 'G:\\Polytech project\\project'
dtypes = {'Title': 'str', 'Text':'str','Category': 'int'}
df = pd.read_csv('G:\\Polytech project\\project\\dataset_reducedother.csv',header=0,names=['Title','Text','Cat'],delimiter=";",dtype=dtypes)
df = df.dropna()
stop = set(stopwords.words("russian"))
df['clean_text'] = df['Text'].apply(remove_punctuation)
df['cut_text'] = df['clean_text'].map(remove_stopwords)


# Установите 50 000 наиболее часто используемых слов
MAX_NB_WORDS = 50000
 # Максимальная длина каждого cut_review
MAX_SEQUENCE_LENGTH = 1000
 # Установите размеры слоя Embeddingceng
EMBEDDING_DIM = 32
 
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['cut_text'].values)
word_index = tokenizer.word_index
print('Всего разных слов:' + str(len(word_index)))
X = tokenizer.texts_to_sequences(df['cut_text'].values)
 # Заполните X, сделайте длину каждого столбца X одинаковой
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
 # Onehot расширение многотипных тегов
Y = pd.get_dummies(df['Cat']).values

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)


model = Sequential()
with tf.device('cpu:0'):
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(350, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_lstm_save_path = 'lstmv3.h5'
    checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path, 
                                        monitor='val_accuracy',
                                        save_best_only=True,
                                        verbose=1)
    print(model.summary())

    epochs = 5
    batch_size = 32
    
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                        callbacks=[checkpoint_callback_lstm, EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])