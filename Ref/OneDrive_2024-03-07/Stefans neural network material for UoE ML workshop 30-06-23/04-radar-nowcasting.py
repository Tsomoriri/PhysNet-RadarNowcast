################################################################################
#
# This code trains a U-Net encoder/decoder neural network for next frame
# prediction of radar images. The input is a sequence of four 40*40 pixel rain
# radar images. The neural network is trained to predict the following (5th)
# image based on the past 4. The data was downloaded from the Met Office Nimrod
# data set, and modified and cleaned up.
#
# (C) Stefan Siegert, s.siegert@exeter.ac.uk
#
################################################################################

import numpy as np
import keras
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# load radar data
movies = np.load('radar_movies.npy')
movies.shape # (980, 40, 40, 20) -- here each movie is of length 20

# in our model we will use the first four images as inputs and predict the
# fifth image
x = movies[:, :, :,  :4]
y = movies[:, :, :, 4:5]


# function: animation of a sequence of radar data (shape = nx,ny,ntime)
def animate(x):
  fig, ax = plt.subplots()
  vmax = np.max(x)
  im = ax.imshow(x[:,:,0], vmin=0, vmax=vmax)
  fig.colorbar(im)
  plt.axis('off')
  def anim_(i):
      im.set_data(x[:,:,i])
      ax.set_title(str(i+1) + '/' + str(x.shape[2]))
  anim = animation.FuncAnimation(
      fig, anim_, interval=300, frames=x.shape[2], repeat_delay=1000)
  plt.show()

# i_plt = 340
# i_plt = 123
i_plt = np.int32(np.random.sample() * movies.shape[0])
animate(x[i_plt,:,:,:])


# train validate test split
tvt = np.tile(['train','train','train','validate','test'], y.shape[0])[:y.shape[0]]
x_train = x[np.where(tvt == 'train')]
y_train = y[np.where(tvt == 'train')]
x_validate = x[np.where(tvt == 'validate')]
y_validate = y[np.where(tvt == 'validate')]
x_test = x[np.where(tvt == 'test')]
y_test = y[np.where(tvt == 'test')]


# plot an input/output pair
i_plt = 20
i_plt = np.int32(np.random.sample() * x_train.shape[0])
for jj in range(4):
  plt.subplot(1,5,jj+1)
  plt.imshow(x_train[i_plt,:,:,jj])
  plt.axis('off')
  plt.title('input')
plt.subplot(1,5,5)
plt.imshow(y_train[i_plt,:,:,0])
plt.title('target output')
plt.axis('off')
plt.show()



###################################################################
# build a U-net encoder/decoder model with skip connections
###################################################################
random.seed(123) # for reproducibility

# input layer
input_shape = x_train.shape[1:]
inputs = keras.layers.Input(input_shape) # (40, 40, 4)

# encoder (3 levels)
conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, 
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(inputs)
conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, 
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(conv1)
pool1 = keras.layers.MaxPool2D(pool_size=(2,2))(conv1)
pool1 = keras.layers.BatchNormalization()(pool1) # (20, 20, 64)

conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, 
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(pool1)
conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, 
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(conv2)
pool2 = keras.layers.MaxPool2D(pool_size=(2,2))(conv2)
pool2 = keras.layers.BatchNormalization()(pool2) # (10, 10, 64)

conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, 
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(pool2)
conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, 
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(conv3)
pool3 = keras.layers.MaxPool2D(pool_size=(2,2))(conv3)
pool3 = keras.layers.BatchNormalization()(pool3) # (5,5,64)


# decoder with skip connections
up4 = keras.layers.UpSampling2D(size=(2,2))(pool3)
up4 = keras.layers.Concatenate(axis=3)([up4, conv3])
conv4 = keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(up4)
conv4 = keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(conv4) # (10, 10, 64)

up5 = keras.layers.UpSampling2D(size=(2,2))(conv4)
up5 = keras.layers.Concatenate(axis=3)([up5, conv2])
conv5 = keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(up5)
conv5 = keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(conv5) # (20, 20, 64)

up6 = keras.layers.UpSampling2D(size=(2,2))(conv5)
up6 = keras.layers.Concatenate(axis=3)([up6, conv1])
conv6 = keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(up6)
conv6 = keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu',
                            kernel_initializer='he_normal')(conv6) # (40, 40, 64)

# output layer
outputs = keras.layers.Conv2D(filters=1, kernel_size=1,
                              padding='same', activation='relu',
                              kernel_initializer='he_normal')(conv6) # (40, 40, 1)

# build and compile
model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()


# fit model and save training and validation loss
history = model.fit(x_train, y_train, epochs=10, batch_size=16, 
                    validation_data = [x_validate, y_validate])

# plot learning curves
plt.plot(history.epoch, history.history['loss'], label='loss')
plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# calculate model output on test data
y_test_pred = model(x_test).numpy()

# calculate test MSE 
np.mean((y_test_pred - y_test)**2)

# as a benchmark, calculate the persistence MSE (use last seen image as
# prediction)
np.mean((x_test[:,:,:,3:] - y_test) ** 2)


# side-by-side animation of two radar movies of shape (nx,ny,ntime) 
def animate2(x1, x2):
  fig, ax = plt.subplots(1,2)
  vmax = np.max(np.append(x1,x2))
  im1 = ax[0].imshow(x1[:,:,0], vmin=0, vmax=vmax)
  im2 = ax[1].imshow(x2[:,:,0], vmin=0, vmax=vmax)
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  fig.colorbar(im2, cax=cbar_ax)
  def anim_(i):
    im1.set_data(x1[:,:,i])
    im2.set_data(x2[:,:,i])
  anim = animation.FuncAnimation(
      fig, anim_, interval=300, frames=x1.shape[2], repeat_delay=1000)
  plt.show()

# i_plt = 90
n_test = x_test.shape[0]
i_plt = np.int32(np.random.sample() * n_test)
print('Showing test case i = ' + str(i_plt))
true = np.append(x_test[i_plt,:,:,:], y_test[i_plt,:,:,:], axis=2)
pred = np.append(x_test[i_plt,:,:,:], y_test_pred[i_plt,:,:,:], axis=2)
animate2(pred, true)


#
# Exercises:
#
# * Why did we choose filters=1 at the output layer, rather than layers=64 like
# in the other Conv2D layers? 
#
# * Transform the image data (x and y) from (40,40) to (37,37) by removing 3
# rows and columns from x and y, and try to run the training code again. Why
# doesn't it work? Which keras layers could be used to address this problem?
#
# * Read about the "EarlyStopping" callback function in keras and implement it.
#
# * How would you extend the model architecture, or your fitted one-step-ahead
# model, to predict several steps ahead instead of just one?
#


