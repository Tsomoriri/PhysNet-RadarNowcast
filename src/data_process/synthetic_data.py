import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
class synData:
    def __init__(self,x,y,t,pde,mux,muy,ictype='gaussian'):
        self.x = x
        self.y = y
        self.t = t
        self.pde = pde
        self.mux = mux
        self.muy = muy
        self.ictype = ictype
        random.seed(23)
    # def u0(self,x):
    #     return np.exp(-100 * (x - 0.2) ** 2)  # gaussian wave
    #

    def u0(self, x, y, n_blobs=5):
        if self.ictype == 'random':
            result = np.zeros_like(x)
            for _ in range(n_blobs):
                center_x = random.random()
                center_y = random.random()
                result += np.exp(-100 * ((x - center_x) ** 2 + (y - center_y) ** 2))
        elif self.ictype == 'normal':
            return np.exp(-100 * (x - 0.2) ** 2) * np.exp(-100 * (y - 0.2) ** 2)  # gaussian wave
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

