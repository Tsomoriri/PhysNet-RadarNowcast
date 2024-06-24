from src.data_process.synthetic_data import synData

import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data
data = synData(100, 100, 2, 0.1, 0.1, 0.1, ictype='random')
indata ,out ,xr,yr= data.generate_training_data()
data.plot_data(xr, yr, out)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

# Create your synData instance
data_generator = synData(x=100, y=100, t=20, pde='advection', mux=0.1, muy=0.15, ictype='random')

# Get the training data
in_data, out_data, xrmesh, yrmesh = data_generator.generate_training_data()

# Extract data for t=0 and flatten
out_data_t0 = out_data[:, :, 0].flatten()

# Apply PowerTransformer (Yeo-Johnson)
pt = PowerTransformer(method='yeo-johnson')
out_data_t0_transformed = pt.fit_transform(out_data_t0.reshape(-1, 1))  # Reshape for 2D input
out_data_t0_transformed = out_data_t0_transformed.flatten()  # Flatten back for analysis

# Analyze the transformed data (t=0)
mean_t0_transformed = np.mean(out_data_t0_transformed)
variance_t0_transformed = np.var(out_data_t0_transformed)

print(f"Mean of transformed data at t=0: {mean_t0_transformed}")
print(f"Variance of transformed data at t=0: {variance_t0_transformed}")

# Plot the distributions (original and transformed) for comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(out_data_t0, bins=30, density=True, alpha=0.7, label='Original Data')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution at t=0 (Original)')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(out_data_t0_transformed, bins=30, density=True, alpha=0.7, label='Transformed Data')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution at t=0 (Power Transformed)')
plt.legend()

plt.show()

import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

# # Create synthetic data
# data = synData(100, 100, 20, 0.1, 0.1, 0.1, ictype='5rect')
# indata ,out ,xr,yr= data.generate_training_data()
# data.plot_data(xr, yr, out)
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PowerTransformer
#
# # Create your synData instance
# data_generator = synData(x=100, y=100, t=20, pde='advection', mux=0.1, muy=0.15, ictype='random')
#
# # Get the training data
# in_data, out_data, xrmesh, yrmesh = data_generator.generate_training_data()
#
# # Extract data for t=0 and flatten
# out_data_t0 = out_data[:, :, 0].flatten()
#
# # Apply PowerTransformer (Yeo-Johnson)
# pt = PowerTransformer(method='yeo-johnson')
# out_data_t0_transformed = pt.fit_transform(out_data_t0.reshape(-1, 1))  # Reshape for 2D input
# out_data_t0_transformed = out_data_t0_transformed.flatten()  # Flatten back for analysis
#
# # Analyze the transformed data (t=0)
# mean_t0_transformed = np.mean(out_data_t0_transformed)
# variance_t0_transformed = np.var(out_data_t0_transformed)
#
# print(f"Mean of transformed data at t=0: {mean_t0_transformed}")
# print(f"Variance of transformed data at t=0: {variance_t0_transformed}")
#
#
#
#
# # Plot the distributions (original and transformed) for comparison
# plt.figure(figsize=(12, 5))
#
# plt.subplot(1, 2, 1)
# plt.hist(out_data_t0, bins=30, density=True, alpha=0.7, label='Original Data')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title('Distribution at t=0 (Original)')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.hist(out_data_t0_transformed, bins=30, density=True, alpha=0.7, label='Transformed Data')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title('Distribution at t=0 (Power Transformed)')
# plt.legend()



#
# plt.show()


# Usage example:
syn_data = synData(x=40, y=40, t=20, pde='advection', mux=0.5, muy=0.5, ictype='rect')

# Generate and save rect movie
rect_movie = syn_data.generate_movie()
syn_data.save_movie(rect_movie, 'rect_movie.npy')

# Generate and save 5rect movie
five_rect_movie = syn_data.generate_movie()
syn_data.save_movie(five_rect_movie, '5rect_movie.npy')

movies = np.load('/home/sushen/PhysNet-RadarNowcast/tests/rect_movie.npy')
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
plt.show()

# train validate test split
tvt = np.tile(['train','train','train','validate','test'], y.shape[0])[:y.shape[0]]
x_train = x[np.where(tvt == 'train')]
y_train = y[np.where(tvt == 'train')]
x_validate = x[np.where(tvt == 'validate')]
y_validate = y[np.where(tvt == 'validate')]
x_test = x[np.where(tvt == 'test')]
y_test = y[np.where(tvt == 'test')]

n_test = x_test.shape[0]
i_plt = np.int32(np.random.sample() * n_test)
true = np.append(x_test[i_plt,:,:,:], y_test[i_plt,:,:,:], axis=2)
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

