Code based on work by @Abayomi Bello on BERT
https://www.researchgate.net/publication/366811050_A_BERT_Framework_to_Sentiment_Analysis_of_Tweets?latestCitations=PB%3A374796188
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import logging
import itertools
import random
import warnings
import time
import os
import datetime as dt
import plotly.express as px
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
import keras
from keras_preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
# Setting some options for general use.


sns.set(font_scale=1.5)
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
warnings.filterwarnings('ignore')
# dataset for model training
model_data = pd.read_csv('news_articles.csv')

model_data.head()
## picking the relevant columns
model_data = model_data[['text_without_stopwords', 'label']]
model_data
## specifying the clean text as  string
model_data.text_without_stopwords=model_data.text_without_stopwords.astype(str)
## to silence warning
os.environ["WANDB_API_KEY"] = "0"
## using the TPU in trainig
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)
## dealing with randomness in results
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)
________________
## splitting to get the development data
train, test = train_test_split(model_data, test_size = 0.20, random_state = 1)
x_train, dev = train_test_split(train, test_size=0.20, random_state=1)
## printing the data shapes
print(x_train.shape)
print(test.shape)
print(dev.shape)
## the label consist of neutral, negative and positive
labels = model_data.label.unique().tolist()
labels
encoder = LabelEncoder()
encoder.fit(model_data.label.tolist())

y_train = encoder.transform(train.label.tolist())
y_test = encoder.transform(test.label.tolist())
y_dev = encoder.transform(dev.label.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_dev = y_dev.reshape(-1,1)

print("y_train",y_train.shape)
print("y_test",y_test.shape)
print("y_dev",y_dev.shape)
# hyperparameters
max_length = 128
batch_size = 128
# Bert Tokenizer
seed(1)
random.set_seed(1)

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
def bert_encode(data):
    seed(1)
    random.set_seed(1)
    tokens = tokenizer.batch_encode_plus(data, max_length=max_length, padding='max_length', truncation=True)

    return tf.constant(tokens['input_ids'])
train_encoded = bert_encode(train.text_without_stopwords)
dev_encoded = bert_encode(dev.text_without_stopwords)
seed(1)
random.set_seed(1)

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_encoded, y_train))
    .shuffle(128)
    .batch(batch_size)
)

dev_dataset = (
    tf.data.Dataset
    .from_tensor_slices((dev_encoded, y_dev))
    .shuffle(128)
    .batch(batch_size)
)
#BERT-BILSTM-FC Model Building
seed(1)
random.set_seed(1)
with strategy.scope():
  bert_encoder = TFBertModel.from_pretrained(model_name)
  input_word_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
  last_hidden_states = bert_encoder(input_word_ids)[0]
  x = tf.keras.layers.SpatialDropout1D(0.2)(last_hidden_states)

  x = tf.keras.layers.Dense(10, activation = 'relu')(x) #2

  x = tf.keras.layers.Dense(10, activation = 'relu')(x) #2

  x = tf.keras.layers.Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)

  x = tf.keras.layers.Dense(128, activation='relu')(x)

  x = tf.keras.layers.Dropout(0.2)(x)

  x = tf.keras.layers.Dense(64, activation='relu')(x)

  x = tf.keras.layers.Dropout(0.3)(x)

  outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

  model = tf.keras.Model(input_word_ids, outputs)

  adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

  model.compile(loss='sparse_categorical_crossentropy',optimizer=adam_optimizer,metrics=['accuracy'])


## model summary
model.summary()
## model plotting
tf.keras.utils.plot_model(model, show_shapes=True)
#Model Training
seed(1)
random.set_seed(1)
history = model.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=10,
    validation_data=dev_dataset, verbose = 1)

## weight saving
model.save_weights('fakenewslabel.h5')
# Evaluation
test_encoded = bert_encode(test.text_without_stopwords)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_encoded)
    .batch(batch_size)
)

predicted_chats = model.predict(test_dataset, batch_size=batch_size)

y_pred = []
for i in range(predicted_chats.shape[0]):
    y_pred.append(np.argmax(predicted_chats[i]))
## classification report
print(classification_report(y_test, y_pred))
## errors
meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

## confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
##plotting the confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
**Prediction**
def decode_sentiment(score):
    if score == 0:
        return "Fake"
    else:
        return "Real"
text = input("Please input your text or review: ")
def predict(text):
    start_at = time.time()
    # Tokenize text
    x_encoded = bert_encode([text])
    # Predict
    score = model.predict([x_encoded])[0]
    # Decode sentiment
    label = decode_sentiment(np.argmax(score))

    return {"label": label, "score": score,
            "elapsed_time": time.time() - start_at}
predict(text)

