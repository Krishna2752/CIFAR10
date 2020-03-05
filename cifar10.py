# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:47:19 2020

@author: KrishnaWork
"""
#Importing necessary libraries.

!pip install tensorflow-gpu==2.0.0-beta1
import numpy as np
np.random.seed(7)

import tensorflow as tf
tf.random.set_seed(7)
from tensorflow import keras
#Importing the CIFAR10 dataset from tensorflow libraries and splitting data into test and training set

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
print("train_images shape:", train_images.shape)
print("test_images shape:", test_images.shape)
print("train_labels shape:", train_labels.shape)
print("test_labels shape:", test_labels.shape)
#Dividing the tensor by 255 to ensure the RGB pixels are in 0 to 1 range
train_images = train_images / 255
test_images = test_images / 255
#Building the convolutional neural network model
model = tf.keras.models.Sequential()
model.add(keras.layers.Conv2D(64, (5,5), activation='relu', input_shape=(32,32,3))),
mdel.add(keras.layers.MaxPooling2D(pool_size=(2, 2))),
model.add(keras.layers.Conv2D(32, (5,5),activation='relu')),
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)),
model.add(keras.layers.Flatten()),
model.add(keras.layers.Dense(64,activation='relu')),
model.add(keras.layers.Dense(10,activation='softmax'))
#compiling the model
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
#fitting the model and setting number of epochs
model.fit(train_images, train_labels, epochs=10, batch_size=128, shuffle=True)
#checking the accuracy
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)



