import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import os
import tqdm
import time
from torch.utils.data import DataLoader, TensorDataset
from src.models.ConvLSTM import ConvLSTM
from src.models.ConvLSTM_Physics import ConvLSTM_iPINN as ConvLSTM_Physics
from src.models.AttentionConvLSTM import ConvLSTM as ConvLSTM_Attention
from src.models.AttentionConvLSTM_Physics import ConvLSTM as ConvLSTM_Attention_Physics

class train_model:
    def __init__(self, model, data, optimizer, loss_fn, device, batch_size, num_epochs):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def update_grid(self,rin_physics):
        # Get the shape of the input tensor
        shape = rin_physics.shape
        # Create an empty tensor with the same shape
        updated_grid = np.zeros(shape)

        # Iterate through each element in the batch
        for i in range(shape[0]):
            # Extract the individual grid
            grid = rin_physics[i]

            # Find the max and min x, y values
            max_x, max_y = np.unravel_index(np.argmax(grid[:, :, 0]), grid[:, :, 0].shape)
            min_x, min_y = np.unravel_index(np.argmin(grid[:, :, 0]), grid[:, :, 0].shape)

            # Set the pattern
            updated_grid[i, max_x, max_y, :] = 1
            updated_grid[i, min_x, min_y, :] = 0

        return updated_grid
    def train(self):
        self.model.to(self.device)


        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in self.data:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(batch_x)
                output = output.squeeze(1)
                output.requires_grad_(True)

                data_loss = self.loss_fn(output, batch_y)
                data_loss.backward(retain_graph=True)
                self.optimizer.step()
                train_loss += data_loss.item() * batch_x.size(0)

        train_loss /= len(self.data.dataset)

    def train_physics(self):
        self.model.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in self.data:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(batch_x)
                output = output.squeeze(1)
                output.requires_grad_(True)

                data_loss = self.loss_fn(output, batch_y)
                rin_physics = torch.zeros_like(batch_x, device=self.device, requires_grad=True)
                output, _ = self.model(rin_physics)
                physics_loss = self.model.advection_loss(rin_physics, output)
                loss = 2*data_loss + physics_loss
                loss.backward(retain_graph=True)
                self.optimizer.step()
                train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(self.data.dataset)


    def train_physics_dynamic_grid(self):
        self.model.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in self.data:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(batch_x)
                output = output.squeeze(1)
                output.requires_grad_(True)

                data_loss = self.loss_fn(output, batch_y)
                rin_physics = torch.zeros_like(batch_x, device=self.device, requires_grad=True)
                rin_physics = self.update_grid(rin_physics.cpu().detach().numpy())
                rin_physics = torch.tensor(rin_physics, dtype=torch.float32, device=self.device, requires_grad=True)
                output, _ = self.model(rin_physics)
                physics_loss = self.model.advection_loss(rin_physics, output)
                loss = 2 * data_loss + physics_loss
                loss.backward(retain_graph=True)
                self.optimizer.step()
                train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(self.data.dataset)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


class dataset_loader:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

    def load_data(self):
        data = np.load(self.path)
        x = data[:, :, :, :4]
        y = data[:, :, :, 4:5]
        # train validate test split
        tvt = np.tile(['train', 'train', 'train', 'validate', 'test'], y.shape[0])[:y.shape[0]]
        x_train = x[np.where(tvt == 'train')]
        y_train = y[np.where(tvt == 'train')]
        train_dataset = TensorDataset(torch.from_numpy(x_train).float().requires_grad_(),
                                      torch.from_numpy(y_train).float())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader




def main():
    # Define hyperparameters
    batch_size = 32
    num_epochs = 1
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define datasets
    datasets = [
        "/home/sushen/PhysNet-RadarNowcast/src/datasets/radar_movies.npy",
    ]

    # Define models
    models = [
        ("ConvLSTM",  ConvLSTM(input_dim=4, hidden_dim=40, kernel_size=(3,3), num_layers=2, physics_kernel_size=(3,3),output_dim=1, batch_first=True),
         ["train"]),
        ("ConvLSTM_Physics", ConvLSTM_Physics(input_dim=4, hidden_dim=40, kernel_size=(3, 3), num_layers=2, output_dim=1, bias=True, return_all_layers=False), ["train_physics", "train_physics_dynamic_grid"]),
        ("ConvLSTM_Attention", ConvLSTM_Attention(input_dim=4, hidden_dim=[128, 64], kernel_size=(3,3), num_layers=2,
                 physics_kernel_size=(3,3), output_dim=1, batch_first=True, bias=True,
                 return_all_layers=False, window_size=1, num_heads=8),
         ["train"]),
        ("ConvLSTM_Attention_Physics", ConvLSTM_Attention_Physics(input_dim=4, hidden_dim=[128, 64], kernel_size=(3,3), num_layers=2,
                 physics_kernel_size=(3,3), output_dim=1, batch_first=True, bias=True,
                 return_all_layers=False, window_size=1, num_heads=8),
         [ "train_physics", "train_physics_dynamic_grid"])
    ]
    # Create a directory for saving models
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    start = time.time()
    # Train each model on each dataset
    for dataset_path in datasets:
        print(f"Training on dataset: {dataset_path}")

        # Extract dataset name from path
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

        # Load the dataset
        data_loader = dataset_loader(dataset_path, batch_size)
        train_loader = data_loader.load_data()

        for model_name, model, training_schemes in models:
            print(f"Training {model_name}")

            # Initialize optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            loss_fn = nn.MSELoss()

            # Create trainer
            trainer = train_model(
                model=model,
                data=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                batch_size=batch_size,
                num_epochs=num_epochs
            )

            # Perform each training scheme
            for scheme in training_schemes:
                print(f"Performing {scheme} training")
                torch.cuda.empty_cache()

                # Reset model and optimizer for each training scheme
                model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                trainer.optimizer = optimizer

                if scheme == "train":
                    trainer.train()
                elif scheme == "train_physics":
                    trainer.train_physics()
                elif scheme == "train_physics_dynamic_grid":
                    trainer.train_physics_dynamic_grid()

                # Save the model
                model_filename = f"{model_name}_{scheme}_{dataset_name}.pth"
                save_path = os.path.join(save_dir, model_filename)
                trainer.save_model(save_path)

    print("Training completed for all models on all datasets.")
    print(f"Total time taken: {time.time() - start} seconds")



if __name__ == "__main__":
    main()















