################################################################################
#
# This code trains a convolutional neural network for image classification. The
# input is a 150*150 pixel photo showing an outdoor scene, labelled as "shine",
# "cloudy", "rain" or "sunrise". The neural network is trained to predict the
# correct label, given an image. The data was downloaded from
# https://data.mendeley.com/datasets/4drtyfjtfy/1 and modified and cleaned a
# bit.
#
# (C) Stefan Siegert, s.siegert@exeter.ac.uk
#
################################################################################


import numpy as np
import keras
import matplotlib.pyplot as plt

# load the saved images
x = np.load('weather-img-x.npy')
y = np.load('weather-img-y.npy')
labls = np.load('weather-img-labls.npy')

# shape of the data sets
x.shape # (1120,150,150,3), 1120 150*150 pixel images in RGB format
y.shape # (1120,4), 1120 labels in one-hot encoding

# plot a random image and its corresponding label
n_img = x.shape[0]
ii = np.int32(np.random.sample() * n_img)
plt.imshow(x[ii])
plt.title(labls[ii])
plt.axis('off')
plt.show()


# train test validate split
tvt = np.tile(['train','train','train','validate','test'], y.shape[0])[:y.shape[0]]
x_train = x[np.where(tvt == 'train')]
y_train = y[np.where(tvt == 'train')]
x_validate = x[np.where(tvt == 'validate')]
y_validate = y[np.where(tvt == 'validate')]
x_test = x[np.where(tvt == 'test')]
y_test = y[np.where(tvt == 'test')]

# build convolutional neural network with several conv2d/max-pooling layers,
# followed by a densely connected layer, and finally a 4-class softmax output
# which is taken as a vector of four probabilities
model = keras.Sequential([
  keras.layers.Input(shape=(150,150,3)),
  keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
  keras.layers.MaxPool2D(pool_size=(2,2)),
  keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
  keras.layers.MaxPool2D(pool_size=(2,2)),
  keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
  keras.layers.MaxPool2D(pool_size=(2,2)),
  keras.layers.Flatten(),
  keras.layers.Dense(100, activation='relu'),
  keras.layers.Dense(4, activation='softmax')
])

# compile and print model summary
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

# train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, 
                    validation_data = [x_validate, y_validate])


# show a random image from test data set, and print the corresponding model
# output and the true label (label order is ['rain', 'cloudy', 'shine',
# 'sunrise'])
def arr_str(x):
  return np.array_str(x, precision=3, suppress_small=True)

ii = np.int32(np.random.sample() * x_test.shape[0])
print(arr_str(model(x_test[ii:(ii+1)]).numpy()))
print(y_test[ii])
plt.imshow(x_test[ii]); plt.show()
plt.show()


#
# Exercises and further reading:
#
# * Simplify the model, e.g. using only one densely connected layer with 4
# outputs and a softmax activation function, and see if you get similar
# accuracy.
#
# * Combine rain+cloudy into one label and shine+sunrise into one label and
# adapt the model code to the binary bad/good weather classification task.
# 
# * Read about "Dropout layers", and retrain the model with 50% Dropout after
# the Flatten layer
# 
# * Read up on the definitions of Categorical Cross Entropy and Accuracy used
# as metrics in this example. What alternatives are there?
# 
# * Plot a confusion matrix for your trained model, using the test data.
#
# * Read about "Image Augmentation layers" in keras. Discuss which image
# transformations are applicable to weather type classification. Refit your
# model with RandomFlip and RandomRotation applied to the input.
#


