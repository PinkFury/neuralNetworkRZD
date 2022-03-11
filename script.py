import keras
import string
import nltk
nltk.download('stopwords')
import tensorflow as tf
from nltk.corpus import stopwords
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU, SpatialDropout1D
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]

    return " ".join(text)

# Count unique words
def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count

# Максимальное количество слов 
num_words = 50000
# Максимальная длина новости
#max_news_len = 87672
max_news_len = 2000
f = open("G:\\Polytech project\\project\\text.txt", "w", encoding = "utf-8")
# Количество классов новостей
nb_classes = 5
dtypes = {'title': 'str', 'text':'str','category': 'int'}
train = pd.read_csv('G:\\Polytech project\\project\\dataset_train.csv', 
                    header=0, 
                    names=['title','text','category'],
                    delimiter=";",dtype=dtypes)
train = train.dropna()
train['title'] = train['title'].map(lambda x: remove_punct(x))
train['text'] = train['title'].map(lambda x: remove_punct(x))
stop = set(stopwords.words("russian"))
train['title'] = train['title'].map(remove_stopwords)
train['text'] = train['text'].map(remove_stopwords)
article = train['title'] + " " + train['text']
#counter = counter_word(text)
#num_words = len(counter)
#print(num_words)
#news = train['title'] + " " + train['text']
#print(news[:5])
y_train = utils.to_categorical(train['category'], nb_classes)
#print(y_train)
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(article)
#print(tokenizer.word_index)
sequences = tokenizer.texts_to_sequences(article)
index = 1
#print(news[index])
#print(sequences[index])
x_train = pad_sequences(sequences, maxlen=max_news_len)
'''
model_gru = Sequential()
model_gru.add(Embedding(num_words, 32, input_length=max_news_len))
model_gru.add(GRU(16))
model_gru.add(Dense(5, activation='softmax'))
model_gru.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model_gru.summary()
model_gru_save_path = 'C:\\Users\\PinkFury\\Desktop\\project\\best_model_gru.h5'
checkpoint_callback_gru = ModelCheckpoint(model_gru_save_path, 
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)
history_gru = model_gru.fit(x_train, 
                              y_train, 
                              epochs=5,
                             batch_size=128,
                            validation_split=0.1,
                           callbacks=[checkpoint_callback_gru])
'''

model_lstm = Sequential()
model_lstm.add(Embedding(num_words, 2100, input_length=max_news_len))
model_lstm.add(LSTM(125,dropout=0.2))
model_lstm.add(Dense(5, activation='softmax'))
optim = tf.keras.optimizers.Adam(learning_rate=3e-4)
model_lstm.compile(optimizer=optim, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model_lstm.summary()
model_lstm_save_path = 'best_model_lstm.h5'
checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path, 
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)
history_lstm = model_lstm.fit(x_train, 
                              y_train, 
                              epochs=4,
                              batch_size=32,
                              validation_split=0.1,
                              callbacks=[checkpoint_callback_lstm], shuffle=True)

test = pd.read_csv('G:\\Polytech project\\project\\dataset_test.csv', 
                    header=0, 
                    names=['title','text','category'],
                    delimiter=";",dtype=dtypes)


test = test.dropna()
test['title'] = test['title'].map(lambda x: remove_punct(x))
test['text'] = test['title'].map(lambda x: remove_punct(x))
stop = set(stopwords.words("russian"))
test['title'] = test['title'].map(remove_stopwords)
test['text'] = test['text'].map(remove_stopwords)
article = test['title'] + " " + test['text']

test_sequences = tokenizer.texts_to_sequences(article)
x_test = pad_sequences(test_sequences, maxlen=max_news_len)
y_test = utils.to_categorical(test['category'], nb_classes)
model_lstm.load_weights(model_lstm_save_path)
model_lstm.evaluate(x_test, y_test, verbose=1)


plt.plot(history_lstm.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history_lstm.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()