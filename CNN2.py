import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from tqdm.notebook import tqdm

from keras.preprocessing import image

import os
c_list = os.listdir('./third/')

lol = image.ImageDataGenerator(validation_split = 0.2)

inp_shape = (154, 217, 3)
inp_lol = inp_shape[:-1]

train = lol.flow_from_directory(
    directory = './third/',
    target_size = inp_lol,
    color_mode = 'rgb',
    batch_size=32,
    class_mode="sparse",
    classes = c_list,
    subset = 'training'
)

test = lol.flow_from_directory(
    directory = './third/',
    target_size = inp_lol,
    color_mode = 'rgb',
    batch_size=32,
    class_mode="sparse",
    classes = c_list,
    subset = 'validation'
)

from keras import layers, models, losses, utils

model = models.Sequential()

model.add(layers.Conv2D(96, (11, 11), strides = (4, 4), input_shape = inp_shape))
model.add(layers.Conv2D(48, (5, 5), strides = (2, 2)))
model.add(layers.Conv2D(24, (2, 2), strides = (1, 1)))
model.add(layers.MaxPool2D((3, 3)))
# model.add(layers.MaxPool2D(()))
model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.Dense(128))
model.add(layers.Dense(len(c_list)))

model.summary()

model.compile(optimizer = 'adam', loss = losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['sparse_categorical_accuracy'])

history = model.fit(train, epochs = 20, validation_data = test)

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