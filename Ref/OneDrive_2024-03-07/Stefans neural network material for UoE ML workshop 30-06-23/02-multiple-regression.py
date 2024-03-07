################################################################################
#
# This code fits simple multiple linear regression model to predict car fuel
# efficiency based on a number of features of the car. Then it extends the
# model into a deep neural network. The code is for educational purposes, to
# get familiar with specifying layers in keras and fitting models. In fact, the
# deep neural network doesn't perform much better than the simple regression
# model.
#
# (C) Stefan Siegert, s.siegert@exeter.ac.uk
#
################################################################################


import keras
import numpy as np
from matplotlib import pyplot as plt
import random

# file 'cars.csv': car efficiency data downloaded from
# https://www.kaggle.com/datasets/beaver68/cars-dataset-in-russia
# columns: Consumption, Seats, Volume, Year, Engine_volume, Maximum_power,
# Cylinders, Speed_to_100, Doors, Width, Length, Height, Volume_fuel_tank,
# Maximum_speed, Full_weight
cars = np.genfromtxt('cars.csv', delimiter=',', skip_header=1)
# 1st column is efficiency (in litres per 100km). We want to build a model of
# efficiency based on the other inputs
y = cars[:,0]
x = cars[:, 1:]

# to monitor overfitting we split into training and validation data 75/25
# train/validate split
tv = np.tile(['train','train','train','validate'], y.shape[0])[:y.shape[0]]
x_train = x[np.where(tv == 'train')]
y_train = y[np.where(tv == 'train')]
x_val = x[np.where(tv == 'validate')]
y_val = y[np.where(tv == 'validate')]

# simple multiple linear regression model in keras (only for illustration,
# otherwise don't do this!)
random.seed(123)
inputs = keras.layers.Input(shape=x.shape[1])
outputs = keras.layers.Dense(1, activation='linear')(inputs)
model1 = keras.Model(inputs, outputs)
model1.compile(optimizer='adam', loss='mse')

# fit the model by providing training data of inputs x and outputs y.
model1.fit(x_train, y_train, batch_size=32, 
          epochs=750, validation_data=[x_val, y_val])

# plot training and validation loss
plt.close()
plt.plot(model1.history.history['loss'], label='training')
plt.plot(model1.history.history['val_loss'], label='validation')
plt.legend()
plt.show(block=False)

# plot predicted values against true consumption values. we see a reasonably
# good fit.
y_val_pred1 = model1(x_val)
plt.plot(y_val, y_val_pred1, 'ok')
plt.axline((0, 0), slope=1, color='k')
plt.show(block=False)

# calculate validation loss by hand
np.mean(keras.losses.mean_squared_error(y_val, y_val_pred1))
np.mean((y_val - y_val_pred1)**2)


# Going DEEEEEP
random.seed(123)
inputs = keras.layers.Input(shape=x.shape[1])
inputs_bn = keras.layers.BatchNormalization()(inputs)
layer1 = keras.layers.Dense(64, activation='relu')(inputs_bn)
layer2 = keras.layers.Dense(32, activation='relu')(layer1)
layer3 = keras.layers.Dense(16, activation='relu')(layer2)
layer4 = keras.layers.Dense(8, activation='relu')(layer3)
outputs = keras.layers.Dense(1, activation='relu')(layer4)
model2 = keras.Model(inputs, outputs)

model2.summary()

model2.compile(optimizer='adam', loss='mse')

# fit the model by providing training data of inputs x and outputs y
model2.fit(x_train, y_train, batch_size=32, 
           epochs=100, validation_data=[x_val, y_val])

# plot training and validation loss
plt.close()
plt.plot(model2.history.history['loss'], label='training')
plt.plot(model2.history.history['val_loss'], label='validation')
plt.legend()
plt.show(block=False)

# reasonably good fit between predicted and true values (although not
# necessarily better than the simple linear regression fit).
y_val_pred2 = model2(x_val)
np.mean((y_val - y_val_pred2)**2)

plt.plot(y_val, y_val_pred2, 'ok')
plt.axline((0, 0), slope=1, color='k')
plt.show(block=False)


np.mean(keras.losses.mean_squared_error(y_val, y_val_pred2))

#
# Exercises and further reading:
#
# * Read up on Stochastic Gradient Descent optimisation and the Adam optimizer.
#
# * Read up on Batch Normalisation.
#
# * Experiment with the number of hidden layers and nodes per layer and see how
# low you can get the validation error.
# 


