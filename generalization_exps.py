import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# Simple sequential model
pre_model = nn.Sequential(
    nn.Linear(2, 1000),
    nn.GELU(),
    nn.Linear(1000, 1000),
    nn.GELU(),
    nn.Linear(1000, 1),
)

def train_model(model, n_epochs=3000, lr=0.0001):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size=32

    for epoch in range(n_epochs):
        # Generate random input
        x = torch.rand(batch_size, 2) * 2 - 1

        # Target is the a Gaussian distribution around the input
        target = torch.exp((-x ** 2).sum(dim=1) / 0.3)
        y = model(x)

        # Only give loss if output is more than 0.5 away from the point (0, 0.5)
        x_dist = x[:, 0]
        y_dist = (x[:, 1] - 0.5)
        dist = torch.sqrt(x_dist ** 2 + y_dist ** 2)
        loss_mask = dist > 0.5

        # Calculate loss
        loss = loss_fn(target=target, input=y[loss_mask])

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: {loss.item()}")


# Graphing the output of pre_model on a 2D space
# by simulating a grid of points and mapping the output
# to a 2D colour space
def graph_output(model):
    # Create grid of points
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(-1, 1, 100)
    xx, yy = torch.meshgrid(x, y)
    grid = torch.stack((xx, yy), dim=2).view(-1, 2)
    
    # Get output of model
    output = model(grid)

    # Map output to 2D colour space using matplotlib
    output = output.view(100, 100, 1)
    # Normalize data to [0,1] range
    output = (output - output.min()) / (output.max() - output.min())
    # Add a third layer of zeros to represent the green channel
    output = torch.cat((output, torch.zeros(100, 100, 2)), dim=2)
    output = output.detach().numpy()


    # Put data in correct shape and format for imshow
    fig, axs = plt.subplots(1, 2)
    circle = plt.Circle((0, 0.5), radius=.05, color="g", fill=True)
    axs[0].add_artist(circle)
    axs[0].imshow(output, extent=[-1, 1, -1, 1])

    # On  the right, plot the target function
    target = torch.exp((-grid ** 2).sum(dim=1) / 0.3)
    target = target.view(100, 100, 1)
    target = (target - target.min()) / (target.max() - target.min())
    target = torch.cat((target, torch.zeros(100, 100, 2)), dim=2)
    target = target.detach().numpy()
    axs[1].imshow(target, extent=[-1, 1, -1, 1])
    plt.show()


if __name__ == "__main__":
    train_model(pre_model)
    graph_output(pre_model)
