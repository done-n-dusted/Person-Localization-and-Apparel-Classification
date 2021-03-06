# -*- coding: utf-8 -*-
"""cnn_project_alex.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GmMGzo6rwYD6mFJBXy8_YmroR2D6FW7F
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from tqdm.notebook import tqdm

from google.colab import drive
drive.mount('/content/gdrive')

from keras.preprocessing import image

import os
c_list = os.listdir('./gdrive/My Drive/cropped/')

lol = image.ImageDataGenerator(validation_split = 0.2)

inp_shape = (154, 217, 3)
inp_lol = inp_shape[:-1]

train = lol.flow_from_directory(
    directory = './gdrive/My Drive/cropped/',
    target_size = inp_lol,
    color_mode = 'rgb',
    batch_size=32,
    class_mode="sparse",
    classes = c_list,
    subset = 'training'
)

test = lol.flow_from_directory(
    directory = './gdrive/My Drive/cropped/',
    target_size = inp_lol,
    color_mode = 'rgb',
    batch_size=32,
    class_mode="sparse",
    classes = c_list,
    subset = 'validation'
)

train.n, test.n

from keras import layers, models, losses, utils

"""Alex Net"""

import tensorflow as tf

model = models.Sequential()

#conv1
model.add(layers.Conv2D(96, (11, 11), strides = (4, 4), input_shape = inp_shape))#, padding = 'same'))
#maxpool1
model.add(layers.MaxPooling2D((3, 3), strides = (2, 2)))
#norm1
model.add(layers.BatchNormalization())

#padding
model.add(layers.ZeroPadding2D((2, 2)))
#conv2
model.add(layers.Conv2D(256, (5, 5), strides = (1, 1)))#, padding = 'same'))
#maxpool2
model.add(layers.MaxPooling2D((3, 3), strides = (2, 2)))
#norm2
model.add(layers.BatchNormalization())

#conv3
model.add(layers.ZeroPadding2D((1, 1)))
model.add(layers.Conv2D(384, (3, 3), strides = (1, 1)))
#conv4
model.add(layers.ZeroPadding2D((1, 1)))
model.add(layers.Conv2D(384, (3, 3), strides = (1, 1)))
#conv5
model.add(layers.ZeroPadding2D((1, 1)))
model.add(layers.Conv2D(384, (3, 3), strides = (1, 1)))#, padding = 'same'))                 

#maxpool3
model.add(layers.MaxPooling2D((3, 3), strides = (2, 2)))

#flatten
model.add(layers.Flatten())

#FCN
model.add(layers.Dense(4096, activation = 'tanh'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096))
model.add(layers.LeakyReLU())
model.add(layers.Dense(len(c_list),activation=tf.nn.softmax))

model.summary()

with tf.device('/device:GPU:0'):
  model.compile(optimizer = 'SGD',
                  loss = losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['sparse_categorical_accuracy'])

  history = model.fit(train, epochs=30, validation_data = test)

plt.figure(figsize = (20, 5))

plt.subplot(121)
# plt.plot(history.history['sparse_categorical_accuracy'], label = 'accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.subplot(122)
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

