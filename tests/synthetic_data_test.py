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
rect_movie = syn_data.generate_ill_movie()
syn_data.save_movie(rect_movie, 'ill_movie.npy')

# Generate and save 3rect movie
syn_data = synData(x=40, y=40, t=20, pde='advection', mux=0.5, muy=0.5, ictype='3rect',n_blobs=3)
rect_movie = syn_data.generate_movie()
syn_data.save_movie(rect_movie, '3rect_movie.npy')
rect_movie = syn_data.generate_ill_movie()
syn_data.save_movie(rect_movie, '3ill_movie.npy')

# Generate and save 11rect movie
syn_data = synData(x=40, y=40, t=20, pde='advection', mux=0.5, muy=0.5, ictype='3rect',n_blobs=11)
rect_movie = syn_data.generate_movie()
syn_data.save_movie(rect_movie, '11rect_movie.npy')
rect_movie = syn_data.generate_ill_movie()
syn_data.save_movie(rect_movie, '11ill_movie.npy')

