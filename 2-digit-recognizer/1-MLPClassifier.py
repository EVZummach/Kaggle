import os
import numpy as np
import pandas as pd

df_train = pd.read_csv('train.csv')
df_train.info()

x = df_train.drop(columns=['label'])
y = df_train['label']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)

from sklearn.neural_network import MLPClassifier, MLPRegressor
clf = MLPClassifier(hidden_layer_sizes=(128, 64, 64), max_iter=50, verbose=True, shuffle=False).fit(x_train, y_train)
clf.score(x_test, y_test)

df_test = pd.read_csv('test.csv')
x_output = scaler.transform(df_test)
y_pred = clf.predict(x_output)

df = pd.DataFrame(y_pred, columns=['Label'])
df.index += 1
df = df.rename_axis('ImageId')
print(df.head)

df.to_csv('submission.csv')