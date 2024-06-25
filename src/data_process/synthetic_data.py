import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
class synData:
    def __init__(self,x,y,t,pde,mux,muy,ictype='gaussian',n_blobs=5):
        self.x = x
        self.y = y
        self.t = t
        self.pde = pde
        self.mux = mux
        self.muy = muy
        self.ictype = ictype
        self.n_blobs = n_blobs
        random.seed(23)
    # def u0(self,x):
    #     return np.exp(-100 * (x - 0.2) ** 2)  # gaussian wave
    #

    def u0(self, x, y):
        if self.ictype == 'random':
            result = np.zeros_like(x)
            for _ in range(self.n_blobs):
                center_x = random.random()
                center_y = random.random()
                result += np.exp(-100 * ((x - center_x) ** 2 + (y - center_y) ** 2))
        elif self.ictype == 'normal':
            return np.exp(-100 * (x - 0.2) ** 2) * np.exp(-100 * (y - 0.2) ** 2)  # gaussian wave

        elif self.ictype == 'rect':
            result = np.zeros_like(x)

            # Define the rectangle boundaries
            x_min, x_max = 0.4, 0.438
            y_min, y_max = 0.4, 0.438

            # Create masks for the rectangle
            x_mask = (x >= x_min) & (x <= x_max)
            y_mask = (y >= y_min) & (y <= y_max)

            # Create gradients within the rectangle
            x_gradient = (x - x_min) / (x_max - x_min)
            y_gradient = (y - y_min) / (y_max - y_min)

            # Combine gradients and apply to the rectangle
            gradient = np.exp(x_gradient * y_gradient)
            result[x_mask & y_mask] = gradient[x_mask & y_mask]

        elif self.ictype == '3rect':
            result = np.zeros_like(x)

            for _ in range(self.n_blobs):
                # Randomly position the rectangle
                x_min = np.random.uniform(0, 0.962)
                y_min = np.random.uniform(0, 0.962)

                # Fixed size for each blob
                width = 0.038
                height = 0.038

                x_max = x_min + width
                y_max = y_min + height

                # Create masks for the rectangle
                x_mask = (x >= x_min) & (x <= x_max)
                y_mask = (y >= y_min) & (y <= y_max)

                # Create gradients within the rectangle
                x_gradient = (x - x_min) / (x_max - x_min)
                y_gradient = (y - y_min) / (y_max - y_min)

                # Combine gradients and apply exponential
                gradient = np.exp(x_gradient * y_gradient)

                # Apply the gradient to the rectangle
                result[x_mask & y_mask] += gradient[x_mask & y_mask]

            # Normalize the result to be between 0 and 1
            if np.max(result) > 0:
                result = result / np.max(result)


        return result
    def generate_training_data(self):
        xr = np.linspace(0, 1, self.x)
        yr = np.linspace(0, 1, self.y)
        tr = np.linspace(0, self.t - 1, self.t).T
        xrmesh, yrmesh, trmesh = np.meshgrid(xr, yr, tr)
        print('xrmesh',xrmesh.shape)
        ur = self.u_2d_true(xrmesh, yrmesh, trmesh)
        print('ur',ur.shape)
        # Stack the 3 2D arrays along a new third dimension, then reshape into a 2D array
        in_data = np.stack((xrmesh, yrmesh, trmesh), axis=-1).reshape(-1, 3)
        in_data = torch.tensor(in_data).float()
        out_data = torch.tensor(ur).float().reshape(-1, 1)

        out_data = out_data.numpy().reshape(self.x, self.y, self.t)
        return in_data, out_data, xrmesh, yrmesh

    def generate_test_data(self):
        xr = np.linspace(0, 1.2, self.x)
        yr = np.linspace(0, 1.2, self.y)
        tr = np.linspace(0, self.t - 1, self.t).T
        xrmesh, yrmesh, trmesh = np.meshgrid(xr, yr, tr)

        test_data = np.stack((xrmesh, yrmesh, trmesh), axis=-1).reshape(-1, 3)
        test_data = torch.tensor(test_data).float()
        return test_data

    # 2D advection equation
    # def u_2d_true(self,x,y,t):
    #     return self.u0(x - self.mux * t) * self.u0(y - self.muy * t)
    def u_2d_true(self, x, y, t):
        return self.u0(x - self.mux * t, y - self.muy * t)

    def plot_data(self, xrmesh, yrmesh, rout_data2):
        plt.contourf(xrmesh[:, :, 0], yrmesh[:, :, 0], rout_data2[:, :, 0])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Initial State with  Blobs (torch) at t=0')
        plt.colorbar()
        plt.colorbar()
        plt.show()

    def generate_movie(self, n_frames=980, nx=40, ny=40, nt=20):
        random.seed(23)
        movie = np.zeros((n_frames, nx, ny, nt))

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 1, nt)
        x_mesh, y_mesh = np.meshgrid(x, y)

        for frame in range(n_frames):
            # Determine which velocity type to use for this frame
            rand_num = random.random()

            if rand_num < 0.4:  # 40% randomly generated velocity
                mux = random.uniform(0, 1)
                muy = random.uniform(0, 1)
            elif rand_num < 0.8:  # 40% constant velocity
                mux = muy = random.uniform(0, 1)
            else:  # 20% exponentially increasing and decreasing velocity
                base_velocity = random.uniform(0, 1)
                mux = muy = lambda t: base_velocity * np.exp(np.sin(2 * np.pi * t))

            for i in range(nt):
                if callable(mux):
                    current_mux = mux(t[i])
                    current_muy = muy(t[i])
                else:
                    current_mux = mux
                    current_muy = muy

                movie[frame, :, :, i] = self.u_2d_true(
                    x_mesh - current_mux * t[i],
                    y_mesh - current_muy * t[i],
                    t[i]
                )

        return movie

    def generate_ill_movie(self, n_frames=980, nx=40, ny=40, nt=20):
        random.seed(23)
        movie = np.zeros((n_frames, nx, ny, nt))

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        t = np.linspace(0, 1, nt)
        x_mesh, y_mesh = np.meshgrid(x, y)

        for frame in range(n_frames):
            # Determine which velocity type to use for this frame
            rand_num = random.random()
            base_velocity = random.uniform(0, 1)
            mux = muy = lambda t: base_velocity * np.exp(np.sin(2 * np.pi * t))

            for i in range(nt):
                if callable(mux):
                    current_mux = mux(t[i])
                    current_muy = muy(t[i])
                else:
                    current_mux = mux
                    current_muy = muy

                movie[frame, :, :, i] = self.u_2d_true(
                    x_mesh - current_mux * t[i],
                    y_mesh - current_muy * t[i],
                    t[i]
                )
        return movie


    def save_movie(self, movie, filename):
        np.save(filename, movie)

    def plot_data(self, xrmesh, yrmesh, rout_data2):

        for t in range(2):
            plt.contourf(xrmesh[:, :, t], yrmesh[:, :, t], rout_data2[:, :, t])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Initial State with  Blobs (torch) at t=0')
            plt.colorbar()
            plt.colorbar()
            plt.show()


