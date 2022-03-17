from nltk.util import pr
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
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
    line = re.sub(r'[^\w\s]','', line) 
    line = re.sub(r"\d+", "", line, flags=re.UNICODE)
    line = re.sub("^\s+|\n|\r|\s+$", '', line)
    return line

def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


#dtypes_start = {'Title': np.string,'Text': string}
#dtypes_finish = {'Title': 'string','Category': 'int'}
#df = pd.read_csv('/var/mkdocs/new_docs/neural_network/htmlData.csv',header=0,delimiter=";")
df = pd.read_csv('G:\\Polytech project\\project\\htmlData_RZD.csv',header=0,delimiter=";")
df_predicted = pd.DataFrame({ 'Title': [], 'Category': [] })
# Установите 50 000 наиболее часто используемых слов
MAX_NB_WORDS = 50000
 # Максимальная длина каждого cut_review
MAX_SEQUENCE_LENGTH = 1000
 # Установите размеры слоя Embeddingceng
EMBEDDING_DIM = 32
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
model = load_model('G:\\Polytech project\\project\\lstmv3.h5')
stop = set(stopwords.words("russian"))
for index, line in df.iterrows():
    line = line.apply(remove_punctuation)
    line = line.map(remove_stopwords)
    seq = tokenizer.texts_to_sequences(line)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    cat_id= pred.argmax(axis=1)[0]
    print(cat_id)
    '''
    tokenizer.fit_on_texts(line['Text'])
    word_index = tokenizer.word_index
    X = tokenizer.texts_to_sequences(line['Text'])
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    #file = open("G:\\Polytech project\\project\\predict_example.txt", "r", encoding='utf-8')
    #txt = file.read()
    model = load_model('G:\\Polytech project\\project\\lstmv2.h5')
    result = model.predict(X)
    print(result[0])
    cat_id= result.argmax(axis=1)[0]
    print(cat_id)
    dic = [line['Title'], cat_id]
    df_predicted.loc[len(df_predicted)] = dic
    print(df_predicted)
    '''
df_predicted.to_csv('predictions.csv', mode='a',index=False, sep=';', encoding='utf-8')