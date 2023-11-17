import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Define the function f[yp(k)] = yp(k) / (1 + yp(k)^2)
def true_function(yp):
    return yp / (1 + yp ** 2)


# Define the neural network with two hidden layers
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # First hidden layer with 10 neurons
        self.fc2 = nn.Linear(10, 10)  # Second hidden layer with 10 neurons
        self.fc3 = nn.Linear(10, 1)  # Output layer

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# Training parameters
num_steps = 10000
learning_rate = 0.0001

# Create the neural network, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Lists to store data for plotting
steps_list = []
loss_list = []
yp_true_list = []
yp_pred_list = []
K = []
# Training loop
for step in range(num_steps):
    # Generate a random input in the interval [-10, 10]
    yp_k = np.random.uniform(-10, 10)
    #yp_k = np.arange(-10,10,0.1)
    K.append(yp_k)

    # Convert the input to a PyTorch tensor
    input_data = torch.tensor([[yp_k]], dtype=torch.float32)

    # Forward pass
    output = model(input_data)

    # Calculate the true function value
    true_value = true_function(yp_k)

    # Calculate the loss
    loss = criterion(output, torch.tensor([[true_value]], dtype=torch.float32))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store data for plotting
    steps_list.append(step)
    loss_list.append(loss.item())
    yp_true_list.append(true_value)
    yp_pred_list.append(output.item())

    # Print the loss every 1000 steps
    if step % 1000== 0:
        print(f'Step {step}, Loss: {loss.item()}')

# Plotting
# plt.scatter(K, yp_true_list)
# plt.scatter(K, yp_pred_list)
for i in range(len(K)):
    for j in range(i, len(K)):
        if K[i] > K[j]:
            t = K[i]
            K[i] = K[j]
            K[j] = t
            t = yp_true_list[i]
            yp_true_list[i] = yp_true_list[j]
            yp_true_list[j] = t
            t = yp_pred_list[i]
            yp_pred_list[i] = yp_pred_list[j]
            yp_pred_list[j] = t
plt.plot(K, yp_true_list)
plt.plot(K, yp_pred_list)

plt.legend()
plt.title('Example 3a Function Approximation')
plt.axis([-10, 10, -1, 1])
plt.show()
# plt.figure(figsize=(10, 6))
#
# # Plot the true function
# plt.plot(steps_list, yp_true_list, label='True Function', linewidth=2)
#
# # Plot the neural network output
# plt.plot(steps_list, yp_pred_list, label='Neural Network Output', linewidth=2)
#
# plt.title('Comparison of True Function and Neural Network Output')
# plt.xlabel('Steps')
# plt.ylabel('Function Value')
# plt.legend()
# plt.show()

# Test the trained model with a few random inputs
for _ in range(5):
    yp_k = np.random.uniform(-10, 10)
    input_data = torch.tensor([[yp_k]], dtype=torch.float32)
    predicted_value = model(input_data).item()
    true_value = true_function(yp_k)
    print(f'Input: {yp_k}, Predicted: {predicted_value}, True: {true_value}')