import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class radarEcho:
    def __init__(self, tn,pn):
        self.tn = tn
        self.pn = pn

    def generate_training_data(self):
        movies = np.load('data/radar_movies.npy')
        # in our model we will use the first tn images as inputs and predict the
        # pn image
        x = movies[:, :, :, :self.tn]
        y = movies[:, :, :, self.tn:self.pn]
        # train validate test split
        tvt = np.tile(['train', 'train', 'train', 'validate', 'test'], y.shape[0])[:y.shape[0]]
        x_train = x[np.where(tvt == 'train')]
        y_train = y[np.where(tvt == 'train')]
        x_validate = x[np.where(tvt == 'validate')]
        y_validate = y[np.where(tvt == 'validate')]
        x_test = x[np.where(tvt == 'test')]
        y_test = y[np.where(tvt == 'test')]


        # Prepare the input and output data
        x_train_tensor = torch.tensor(x_train).float()
        y_train_tensor = torch.tensor(y_train).float()
        x_test_tensor = torch.tensor(x_test).float()
        y_test_tensor = torch.tensor(y_test).float()

        # Create a TensorDataset
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        # Create a DataLoader
        batch_size = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

