import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.metrics import structural_similarity as ssim
from src.models.ConvLSTM import ConvLSTM
from src.models.ConvLSTM_Physics import ConvLSTM_iPINN as ConvLSTM_Physics
from src.models.AttentionConvLSTM import ConvLSTM as ConvLSTM_Attention
from src.models.AttentionConvLSTM_Physics import ConvLSTM as ConvLSTM_Attention_Physics


class ModelEvaluator:
    def __init__(self, models_dir, datasets_dir, batch_size=32, device='cuda'):
        self.models_dir = models_dir
        self.datasets_dir = datasets_dir
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results_dir = os.path.join(os.path.dirname(__file__), 'src', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self, path):
        data = np.load(path)
        x = data[:, :, :, :4]
        y = data[:, :, :, 4:5]
        tvt = np.tile(['train', 'train', 'train', 'validate', 'test'], y.shape[0])[:y.shape[0]]
        x_test = x[np.where(tvt == 'test')]
        y_test = y[np.where(tvt == 'test')]
        test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    # def load_model(self, path):
    #     model_name = os.path.basename(path).split('_')[0]
    #
    #     if model_name == 'ConvLSTM_Attention_Physics':
    #         model = ConvLSTM_Attention_Physics(input_dim=4, hidden_dim=[128, 64], kernel_size=(3, 3), num_layers=2,
    #                                            physics_kernel_size=(3, 3), output_dim=1, batch_first=True, bias=True,
    #                                            return_all_layers=False, window_size=1, num_heads=8)
    #     elif model_name == 'ConvLSTM_Attention':
    #         model = ConvLSTM_Attention(input_dim=4, hidden_dim=[128, 64], kernel_size=(3, 3), num_layers=2,
    #                                    physics_kernel_size=(3, 3), output_dim=1, batch_first=True, bias=True,
    #                                    return_all_layers=False, window_size=1, num_heads=8)
    #     elif model_name == 'ConvLSTM_Physics':
    #         model = ConvLSTM_Physics(input_dim=4, hidden_dim=40, kernel_size=(3, 3), num_layers=2, output_dim=1,
    #                                  bias=True, return_all_layers=False)
    #     elif model_name == 'ConvLSTM':
    #         model = ConvLSTM(input_dim=4, hidden_dim=40, kernel_size=(3, 3), num_layers=2, physics_kernel_size=(3, 3),
    #                          output_dim=1, batch_first=True)
    #     else:
    #         raise ValueError(f"Unknown model type: {model_name}")
    #
    #     model.load_state_dict(torch.load(path, map_location=self.device))
    #     return model.to(self.device)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        model_name = os.path.basename(path).split('_')[0]

        # Determine model parameters from the state dict
        if 'cell_list.0.conv.weight' in state_dict:
            input_dim = state_dict['cell_list.0.conv.weight'].shape[1] - state_dict['cell_list.0.conv.weight'].shape[0] // 4
            hidden_dim = state_dict['cell_list.0.conv.weight'].shape[0] // 4
        else:
            raise ValueError(f"Unable to determine model parameters from state dict for {path}")

        num_layers = sum(1 for key in state_dict if 'cell_list' in key and 'conv.weight' in key)
        kernel_size = state_dict['cell_list.0.conv.weight'].shape[2:]
        output_dim = state_dict['output_conv.weight'].shape[0]

        # Create the appropriate model
        if model_name == 'ConvLSTM':
            model = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                             num_layers=num_layers, physics_kernel_size=kernel_size, output_dim=output_dim,
                             batch_first=True)
        elif model_name == 'ConvLSTM_Physics':
            model = ConvLSTM_Physics(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                                     num_layers=num_layers, output_dim=output_dim, bias=True,
                                     return_all_layers=False)
        elif model_name in ['ConvLSTM_Attention', 'ConvLSTM_Attention_Physics']:
            hidden_dims = [hidden_dim] * num_layers
            ModelClass = ConvLSTM_Attention if model_name == 'ConvLSTM_Attention' else ConvLSTM_Attention_Physics
            model = ModelClass(input_dim=input_dim, hidden_dim=hidden_dims, kernel_size=kernel_size,
                               num_layers=num_layers, physics_kernel_size=kernel_size, output_dim=output_dim,
                               batch_first=True, bias=True, return_all_layers=False, window_size=1, num_heads=8)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Load the state dict, ignoring mismatched keys
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model.to(self.device)

    def calculate_metrics(self, outputs, targets):
        mae = np.mean(np.abs(outputs - targets))

        # Determine the appropriate window size for SSIM
        min_dim = min(outputs.shape[-2:])
        win_size = min(7, min_dim)  # Use 7 or the smallest dimension, whichever is smaller
        if win_size % 2 == 0:
            win_size -= 1  # Ensure odd window size

        try:
            ssim_value = ssim(outputs.squeeze(), targets.squeeze(),
                              data_range=targets.max() - targets.min(),
                              win_size=win_size,
                              channel_axis=None if outputs.squeeze().ndim == 2 else 0)
        except ZeroDivisionError:
            # Fallback to normalized MSE if SSIM fails
            mse = np.mean((outputs - targets) ** 2)
            data_range = targets.max() - targets.min()
            if data_range != 0:
                ssim_value = 1 - mse / (data_range ** 2)  # Normalized MSE, inverted to be similar to SSIM
            else:
                ssim_value = 1.0  # Perfect similarity if there's no variation in the target

        return mae, ssim_value

    def animate_comparison(self, outputs, targets, filename):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        def update(i):
            ax[0].clear()
            ax[1].clear()
            ax[0].imshow(outputs[i].squeeze(), cmap='gray')
            ax[0].set_title('Output')
            ax[1].imshow(targets[i].squeeze(), cmap='gray')
            ax[1].set_title('Target')

        anim = FuncAnimation(fig, update, frames=len(outputs), interval=200)
        anim.save(filename, writer='imagemagick')
        plt.close(fig)

    def evaluate_model(self, model_path, dataset_path):
        model = self.load_model(model_path)
        model.eval()
        data_loader = self.load_data(dataset_path)

        all_outputs = []
        all_targets = []
        mae_sum = 0
        ssim_sum = 0
        n_samples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs)
                outputs = outputs.cpu().numpy()
                targets = targets.numpy()

                mae, ssim_value = self.calculate_metrics(outputs, targets)
                mae_sum += mae * len(inputs)
                ssim_sum += ssim_value * len(inputs)
                n_samples += len(inputs)

                all_outputs.append(outputs)
                all_targets.append(targets)

        avg_mae = mae_sum / n_samples
        avg_ssim = ssim_sum / n_samples

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        return avg_mae, avg_ssim, all_outputs, all_targets

    def run_experiment(self, experiment_name, model_filter, dataset_filter):
        experiment_results = []
        for model_file in os.listdir(self.models_dir):
            if model_filter(model_file):
                model_path = os.path.join(self.models_dir, model_file)
                for dataset_file in os.listdir(self.datasets_dir):
                    if dataset_filter(dataset_file):
                        dataset_path = os.path.join(self.datasets_dir, dataset_file)

                        avg_mae, avg_ssim, outputs, targets = self.evaluate_model(model_path, dataset_path)

                        result = {
                            'model': model_file,
                            'dataset': dataset_file,
                            'mae': avg_mae,
                            'ssim': avg_ssim
                        }
                        experiment_results.append(result)

                        # Save comparison GIF
                        gif_file = os.path.join(self.results_dir,
                                                f'{experiment_name}_{model_file}_{dataset_file}_comparison.gif')
                        self.animate_comparison(outputs, targets, gif_file)

        # Save experiment results
        result_file = os.path.join(self.results_dir, f'{experiment_name}_results.txt')
        with open(result_file, 'w') as f:
            for result in experiment_results:
                f.write(f"Model: {result['model']}\n")
                f.write(f"Dataset: {result['dataset']}\n")
                f.write(f"MAE: {result['mae']}\n")
                f.write(f"SSIM: {result['ssim']}\n\n")

        print(f"Experiment '{experiment_name}' completed. Results saved in {result_file}")

    def run_all_experiments(self):
        # Experiment 1: Effect of dynamic grid
        self.run_experiment('dynamic_grid',
                            lambda m: 'dynamic_grid' in m,
                            lambda d: True)

        # Experiment 2: Physics models
        self.run_experiment('physics',
                            lambda m: 'Physics' in m and 'dynamic_grid' not in m,
                            lambda d: True)

        # Experiment 3: Increasing particle effect (rect datasets)
        self.run_experiment('increasing_particles',
                            lambda m: True,
                            lambda d: d in ['rect_movie.npy', '3rect_movie.npy', '11rect_movie.npy'])

        # Experiment 4: Ill-posed exponential increase in velocity
        self.run_experiment('ill_posed_velocity',
                            lambda m: True,
                            lambda d: d in ['ill_movie.npy', '3ill_movie.npy', '11ill_movie.npy'])


# Example usage
if __name__ == "__main__":
    models_dir = "/home/sushen/PhysNet-RadarNowcast/src/saved_models"
    datasets_dir = "/home/sushen/PhysNet-RadarNowcast/src/datasets"

    evaluator = ModelEvaluator(models_dir, datasets_dir)
    evaluator.run_all_experiments()