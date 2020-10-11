import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout


df = pd.read_csv('Dataset/data_train.csv')
df = df.dropna()


X = df.drop('Emotion', axis = 1)
Y = df['Emotion']
print(X)
print(Y)

print(Y.value_counts())

messages = X.copy()

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
Y = label_encoder.fit_transform(Y)

print(Y)
# 0 - anger
# 1 - fear
# 2 - joy
# 3 - neutral
# 4 - sadness

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []

#removing unnecessary information from data
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['Text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#converting data into one hot vectors
voc_size = 10000    
onehot_repr=[one_hot(words,voc_size)for words in corpus] 

sent_length=250
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

#model
embedded_vector_features = 200
model = Sequential()
model.add(Embedding(voc_size, embedded_vector_features, input_length=sent_length))
model.add(Bidirectional(LSTM(200)))
model.add(Dropout(0.3))
model.add(Dense(5, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

import numpy as np
X_final=np.array(embedded_docs)
Y_final=np.array(Y)

print(X_final.shape,Y_final.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.33, random_state=42)

#splitting data for training and validation
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 10, batch_size=64)

Y_pred=model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,Y_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_pred))

from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))

model.save('twitter_sentiment_analysis_LSTM.h5')

