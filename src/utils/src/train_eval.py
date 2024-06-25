import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import your model classes here
from src.models.ConvLSTM import ConvLSTM
from src.models.ConvLSTM_Physics import ConvLSTM_iPINN as ConvLSTM_Physics
from src.models.AttentionConvLSTM import ConvLSTM as ConvLSTM_Attention
from src.models.AttentionConvLSTM_Physics import ConvLSTM as ConvLSTM_Attention_Physics

class TrainEvalManager:
    def __init__(self, models_config, datasets_config, device='cuda', batch_size=32, num_epochs=50, learning_rate=0.001):
        self.models_config = models_config
        self.datasets_config = datasets_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.results_dir = os.path.join(os.path.dirname(__file__), 'Train_results')
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self, path):
        data = np.load(path)
        x = data[:, :, :, :4]
        y = data[:, :, :, 4:5]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def train_model(self, model, train_loader, optimizer, loss_fn, scheme):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            output, _ = model(batch_x)
            output = output.squeeze(1)

            if scheme == 'standard':
                loss = loss_fn(output, batch_y)
            elif scheme in ['physics', 'physics_dynamic_grid']:
                data_loss = loss_fn(output, batch_y)
                rin_physics = torch.zeros_like(batch_x, device=self.device, requires_grad=True)
                if scheme == 'physics_dynamic_grid':
                    rin_physics = self.update_grid(rin_physics.cpu().detach().numpy())
                    rin_physics = torch.tensor(rin_physics, dtype=torch.float32, device=self.device, requires_grad=True)
                physics_output, _ = model(rin_physics)
                physics_loss = model.advection_loss(rin_physics, physics_output)
                loss = 2 * data_loss + physics_loss
            else:
                raise ValueError(f"Unknown training scheme: {scheme}")

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        return total_loss / len(train_loader.dataset)

    def evaluate_model(self, model, test_loader):
        model.eval()
        total_loss = 0.0
        mae_sum = 0.0
        ssim_sum = 0.0
        n_samples = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output, _ = model(batch_x)
                output = output.squeeze(1)

                loss = nn.MSELoss()(output, batch_y)
                total_loss += loss.item() * batch_x.size(0)

                output_np = output.cpu().numpy()
                batch_y_np = batch_y.cpu().numpy()

                mae_sum += np.sum(np.abs(output_np - batch_y_np))
                for i in range(output_np.shape[0]):
                    ssim_value = ssim(output_np[i].squeeze(), batch_y_np[i].squeeze(), data_range=batch_y_np[i].max() - batch_y_np[i].min())
                    ssim_sum += ssim_value
                n_samples += batch_x.size(0)

                all_outputs.append(output_np)
                all_targets.append(batch_y_np)

        avg_loss = total_loss / n_samples
        avg_mae = mae_sum / n_samples
        avg_ssim = ssim_sum / n_samples

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        return avg_loss, avg_mae, avg_ssim, all_outputs, all_targets

    def update_grid(self, rin_physics):
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
    def create_comparison_gif(self, outputs, targets, filename):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        def update(i):
            ax[0].clear()
            ax[1].clear()
            ax[0].imshow(outputs[i].squeeze(), cmap='viridis')
            ax[0].set_title('Model Output')
            ax[1].imshow(targets[i].squeeze(), cmap='viridis')
            ax[1].set_title('Ground Truth')

        anim = FuncAnimation(fig, update, frames=min(len(outputs), 100), interval=200)  # Limit to 100 frames
        anim.save(filename, writer='pillow', fps=5)
        print(f"Comparison GIF saved as {filename}")
        plt.close(fig)

    def run_experiment(self, model_config, dataset_path, experiment_name):
        model_name, model_class, model_params, training_schemes = model_config
        train_loader, test_loader = self.load_data(dataset_path)

        results = []

        for scheme in training_schemes:
            print(f"Model Parameters: {model_params}")
            model = model_class(**model_params).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_fn = nn.MSELoss()

            print(f"Training {model_name} with {scheme} scheme on {dataset_path}")
            for epoch in range(self.num_epochs):
                train_loss = self.train_model(model, train_loader, optimizer, loss_fn, scheme)
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {train_loss:.4f}")

            # Evaluate immediately after training
            test_loss, mae, ssim_value, outputs, targets = self.evaluate_model(model, test_loader)
            print(f"Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, SSIM: {ssim_value:.4f}")

            results.append({
                'model': model_name,
                'scheme': scheme,
                'dataset': os.path.basename(dataset_path),
                'test_loss': test_loss,
                'mae': mae,
                'ssim': ssim_value
            })

            # Create and save comparison GIF
            gif_filename = os.path.join(self.results_dir, f'{experiment_name}_{model_name}_{scheme}_comparison.gif')
            self.create_comparison_gif(outputs, targets, gif_filename)
            print(f"Comparison GIF saved as {gif_filename}")

        # Save experiment results
        result_file = os.path.join(self.results_dir, f'{experiment_name}_results.txt')
        with open(result_file, 'w') as f:
            for result in results:
                f.write(f"Model: {result['model']}\n")
                f.write(f"Scheme: {result['scheme']}\n")
                f.write(f"Dataset: {result['dataset']}\n")
                f.write(f"Test Loss: {result['test_loss']}\n")
                f.write(f"MAE: {result['mae']}\n")
                f.write(f"SSIM: {result['ssim']}\n\n")

        print(f"Experiment '{experiment_name}' completed. Results saved in {result_file}")

    def run_all_experiments(self):
        for dataset_config in self.datasets_config:
            dataset_path, experiment_name = dataset_config
            for model_config in self.models_config:
                self.run_experiment(model_config, dataset_path, experiment_name)

def main():
    # Define your models
    models_config = [
        ("ConvLSTM", ConvLSTM,
         {"input_dim": 4, "hidden_dim": 40, "kernel_size": (3,3),"num_layers": 2, "physics_kernel_size": (3,3), "output_dim": 1, "batch_first": True},
         ["standard"]),
        ("ConvLSTM_Physics", ConvLSTM_Physics,
         {"input_dim": 4, "hidden_dim": 40, "kernel_size": (3,3), "num_layers": 2, "output_dim": 1, "bias": True, "return_all_layers": False},
         ["physics", "physics_dynamic_grid"]),
        ("ConvLSTM_Attention", ConvLSTM_Attention,
         {"input_dim": 4, "hidden_dim": [128, 64], "kernel_size": (3,3), "physics_kernel_size": (3,3), "num_layers": 2, "output_dim": 1, "batch_first": True, "bias": True, "return_all_layers": False, "window_size": 1, "num_heads": 8},
         ["standard"]),
        ("ConvLSTM_Attention_Physics", ConvLSTM_Attention_Physics,
         {"input_dim": 4, "hidden_dim": [128, 64], "kernel_size": (3,3), "physics_kernel_size": (3,3), "num_layers": 2, "output_dim": 1, "batch_first": True, "bias": True, "return_all_layers": False, "window_size": 1, "num_heads": 8},
         ["physics", "physics_dynamic_grid"])
    ]

    # Define your datasets
    datasets_config = [
        ("/home/sushen/PhysNet-RadarNowcast/src/datasets/rect_movie.npy", "Rectangle_whole"),
        ("/home/sushen/PhysNet-RadarNowcast/src/datasets/radar_movies.npy", "Radar_data"),
        # Add more datasets as needed
    ]

    manager = TrainEvalManager(models_config, datasets_config)
    manager.run_all_experiments()

if __name__ == "__main__":
    main()