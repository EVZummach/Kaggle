import os
import numpy as np
import pandas as pd

df_train = pd.read_csv(r'train.csv')
df_train.info()

x = df_train.drop(columns=['label'])
y = pd.get_dummies(df_train['label'])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout

model = Sequential()

model.add(Dense(128, activation='tanh', input_shape=(x_train.shape[1], ), kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(64, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs = 20, verbose=1, validation_data=(x_test, y_test))

df_test = pd.read_csv('test.csv')
x_output = scaler.transform(df_test)
y_pred = np.argmax(model.predict(x_output), axis=1).reshape(-1, 1)

df = pd.DataFrame(y_pred, columns=['Label'])
df.index += 1
df = df.rename_axis('ImageId')
print(df.head)

df.to_csv('submission.csv')