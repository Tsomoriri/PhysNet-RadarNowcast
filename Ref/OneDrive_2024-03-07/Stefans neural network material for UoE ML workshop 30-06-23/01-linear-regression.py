################################################################################
#
# This code simulates a data set of x and y values by a linear relationship y =
# 2 + 0.5 * x + noise. Then it uses tensorflow/keras to learn the parameters 2
# and 0.5 from the data. Note that this example is purely educational, and you
# should *not* use keras to fit a linear regression model!
#
# (C) Stefan Siegert, s.siegert@exeter.ac.uk
#
################################################################################


# load libraries
import keras
import numpy as np
from matplotlib import pyplot as plt

# generate random training data (scalar input x, scalar output y, noisy linear
# relationship)
n = 100
x = np.random.normal(size=n)
y = 2 + 0.5 * x + 0.3 * np.random.normal(size=n)
plt.plot(x, y, 'ok')
plt.show()

# define the linear regression model in keras (input layer accepts inputs of
# dimension 1, output is calculated in a "dense" layer, i.e. linear
# transformation of input, and the linear activation function f(x)=x is applied
# to produce the output)
inputs = keras.layers.Input(shape=(1))
outputs = keras.layers.Dense(1, activation='linear')(inputs)
model = keras.Model(inputs, outputs)

# compile the model, i.e. specify the optimisation algorithm and the loss
# function
model.compile(optimizer='adam', loss='mse')

# fit the model by providing training data of inputs x and outputs y.
model.fit(x, y, batch_size=20, epochs=1000)

# calculate fitted values
y_pred = model(x)
xn_true = np.arange(-3,3,.1)
yn_true = 2 + 0.5 * xn_true

# the model was learned correctly
plt.plot(x, y_pred, 'ok')
plt.plot(xn_true, yn_true, '-b')
plt.show(block=False)

# show the learned weights (should be close to 2 and 0.5 as used when
# simulating the data)
model.summary()
model.layers[0].get_weights()
model.layers[1].get_weights()


