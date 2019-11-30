#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:37:42 2019

@author: berk
"""
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
        results = np.zeros((len(sequences),dimension))
        for i, sequence in enumerate(sequences):
                results[i,sequence]= 1.
        return results


#import dataset from keras
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#prepare the data for the network
max([max(sequence) for sequence in train_data])
word_index= imdb.get_word_index()
reverse_word_index= dict([(value,key) for (key, value) in word_index.items()])
decoded_review= ''.join([reverse_word_index.get(i-3,"?")for i in train_data[0]])
x_train = vectorize_sequences(train_data)
x_test  = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype("float32")
y_test  = np.asarray(test_labels).astype("float32")


#initial the network

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation="relu",input_shape=(10000,)))
model.add(layers.Dense(16,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])


#initial validation dataset
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#train model
model.fit(partial_x_train,
          partial_y_train,
          epochs=3,  
          batch_size=512,
          validation_data=(x_val,y_val))

#visualation of train process
import matplotlib.pyplot as plt

history_dict = model.history.history
loss_value = history_dict["loss"]
val_loss_values = history_dict["val_loss"]

epochs = range(1,len(loss_value)+1)


plt.plot(epochs,loss_value,"bo",label="Train loss")
plt.plot(epochs, val_loss_values,"b" , label="acc loss")
plt.title("Train and acc loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()









