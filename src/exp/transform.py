from src.data_process.synthetic_data import synData
from src.models.FCN import FCN_iPINN
from sklearn.preprocessing import PowerTransformer
import numpy as np
import matplotlib.pyplot as plt
import torch


data_generator = synData(x=100, y=100, t=20, pde='advection', mux=0.1, muy=0.15, ictype='normal')

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

torch.manual_seed(123)
# pinn = FCN(3,1,32,3)
pinn = FCN_iPINN(3, 1, 32, 3)
# mu = torch.tensor(np.linspace(0.1, 0.5, 100).reshape(100,1)).float()

# mu_iPINN = torch.nn.Parameter(torch.tensor([1.5], requires_grad=True))
mu_lr = 1e-1
print("The true value of Vx is: ", mux, "\n", "The initial guessed value of Vx is: ", pinn.velocity_x.item())
print("The true value of Vy is: ", muy, "\n", "The initial guessed value of Vy is: ", pinn.velocity_y.item())
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-4)

for i in range(5000):
    optimizer.zero_grad()

    # compute the "data loss"
    uh = pinn(rin_data)
    loss1 = torch.mean((uh - rout_data) ** 2)  # use mean squared error

    # compute the "physics loss"
    uhp = pinn(rin_physics)
    grad = torch.autograd.grad(outputs=uhp, inputs=rin_physics,
                               grad_outputs=torch.ones_like(uhp), create_graph=True)[0]
    dudx = grad[:, 0]
    dudy = grad[:, 1]
    dudt = grad[:, 2]
    physics = dudt + pinn.velocity_x * dudx + pinn.velocity_y * dudy  # this term is zero if the model output satisfies the advection equation
    loss2 = torch.mean((physics) ** 2)

    # backpropagate combined loss
    loss = 2 * loss1 + loss2  # add two loss terms together

    dJdvx = torch.autograd.grad(loss, pinn.velocity_x, retain_graph=True, create_graph=True)[0]
    pinn.velocity_x.data = pinn.velocity_x.data - (mu_lr * dJdvx)

    dJdvy = torch.autograd.grad(loss, pinn.velocity_y, retain_graph=True, create_graph=True)[0]
    pinn.velocity_y.data = pinn.velocity_y.data - (mu_lr * dJdvy)

    loss.backward(retain_graph=True)
    optimizer.step()

    if (i + 1) % 1000 == 0:
        print(f'Epoch: {i + 1}/{10000}, Loss: {loss.item()}')
        print("At itteration: ", i, "\n", "The current value of Vx is: ", pinn.velocity_x.item())
        print("At itteration: ", i, "\n", "The current value of Vy is: ", pinn.velocity_y.item())
        print(f' NN loss : {loss1.item()}, Physics loss : {loss2.item()}')