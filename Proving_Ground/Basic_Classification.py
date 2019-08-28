import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

# here are 4 numpy arrays from the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# the images are 28*28 Numpy arrays

print(train_images)

# the labels
print(train_labels)

# look for the shape of the format of trained images
# 60000 images, two dimensions: 28*28
print(train_images.shape)

# how many images in the training set and testing set
len(train_images)
len(test_images)

# show the first image in the training set
plt.figure()

# the first images
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# normalization
train_images = train_images / 255.0
test_images = test_images / 255.0

# create a class names because it didn't
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# display 25 figures in the training set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Construct the network layer
model = keras.Sequential([
    # transform the two-dimension array into a one-dimension array, that may the reason why it called 'flatten'
    keras.layers.Flatten(input_shape=(28, 28)),

    # the network is constructed by two networking layer
    # the first layer has 128 nodes, the second layer has 10 nodes
    # the first layer called relu and the second layer called softmax
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# parameters, including optimizer, loss function and evaluation methods
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the models:
model.fit(train_images, train_labels, epochs=5)

# predict all elements in the test sets
predictions = model.predict(test_images)

# these float are the "confident" of the prediction
print(predictions[0])

# the largest one
print(np.argmax(predictions[0]))

# check whether the label are right
print(test_labels[0])