import os
import numpy as np
import torch

class synData:
    def __init__(self,x,y,t,pde,mux,muy,u0):
        self.x = x
        self.y = y
        self.t = t
        self.pde = pde
        self.mux = mux
        self.muy = muy
        self.u0 = u0

    def u0(self):
        return np.exp(-100 * (self.x - 0.2) ** 2)  # gaussian wave

    def generate_training_data(self):
        xr = np.linspace(0, 1, self.x)
        yr = np.linspace(0, 1, self.y)
        tr = np.linspace(0, self.t - 1, self.t).T
        xrmesh, yrmesh, trmesh = np.meshgrid(xr, yr, tr)
        ur = self.u_2d_true(xrmesh, yrmesh, trmesh)
        # Stack the 3 2D arrays along a new third dimension, then reshape into a 2D array
        in_data = np.stack((xrmesh, yrmesh, trmesh), axis=-1).reshape(-1, 3)
        in_data = torch.tensor(in_data).float()
        out_data = torch.tensor(ur).float().reshape(-1, 1)

        out_data = out_data.numpy().reshape(self.x, self.y, self.t)
        return in_data, out_data

    def generate_test_data(self):
        xr = np.linspace(0, 1.2, self.x)
        yr = np.linspace(0, 1.2, self.y)
        tr = np.linspace(0, self.t - 1, self.t).T
        xrmesh, yrmesh, trmesh = np.meshgrid(xr, yr, tr)

        test_data = np.stack((xrmesh, yrmesh, trmesh), axis=-1).reshape(-1, 3)
        test_data = torch.tensor(test_data).float()
        return test_data

    # 2D advection equation
    def u_2d_true(self,x,y,t):
        return self.u0(x - self.mux * t) * self.u0(y - self.muy * t)

