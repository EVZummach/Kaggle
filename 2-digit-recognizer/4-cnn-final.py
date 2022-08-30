import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv(r'train.csv')
df_train.info()

from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()


x_input = df_train.drop(columns=['label']).values
x = mm_scaler.fit_transform(x_input).reshape(-1, 28, 28)
y = pd.get_dummies(df_train['label'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test))