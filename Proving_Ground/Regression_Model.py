import numpy as np
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("lrTest.csv")

X = df.iloc[:,0:2]
X = np.array(X)
X = X.astype(float)
X = X/20
print("dataframe of X is: ")
print(X)


Y = df.iloc[:,2:3]
Y = np.array(Y)
Y = Y.astype(float)
Y = Y/20
print("dataframe of Y is: ")
print(Y)

# get the liens before 16 as training set
X_train, Y_train = X[:15], Y[:15]
print("dataframe of x_train is: ")
print(X_train)

print("dataframe of y_train is: ")
print(Y_train)

# get the line after 16 as testing set
X_test, Y_test = X[15:],Y[15:]
print("dataframe of x_test is: ")
print(X_test)

print("dataframe of y_test is: ")
print(Y_test)

# construct the models
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(64, activation=tf.nn.relu),
     tf.keras.layers.Dense(64, activation=tf.nn.relu),
     tf.keras.layers.Dense(1)
     ])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])

# epochs: iteration times
# batch_size: divide the data into batch and train these batch during the training process
model.fit(X_train,Y_train,epochs=5,batch_size=15)

prediction = model.predict(X_test, batch_size = 1)

print(prediction*20)