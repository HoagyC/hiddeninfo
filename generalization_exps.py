import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# Simple sequential model
pre_width=100

class BaseModel(nn.Module):
    def __init__(self, gain=1.):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(10, pre_width),
            nn.GELU(),
            nn.Linear(pre_width, pre_width),
            nn.GELU(),
            nn.Linear(pre_width, 10),
        )
        self.model = model
        meta_init(self.model, gain=1.)
    
    def forward(self, x):
        return self.model(x)

# Simple sequential model
true_width=100
bottleneck_width=10
true_pre_model = nn.Sequential(
    nn.Linear(10, true_width),
    nn.ReLU(),
    nn.Linear(true_width, true_width),
    nn.ReLU(),
    nn.Linear(true_width, true_width),
    nn.ReLU(),
    nn.Linear(true_width, bottleneck_width),
)
# Turn off requires_grad for true_pre_model
for param in true_pre_model.parameters():
    param.requires_grad = False

true_post_model = nn.Sequential(
    nn.Linear(bottleneck_width, 10),
)

# Turn off requires_grad for true_post_model
for param in true_post_model.parameters():
    param.requires_grad = False

def meta_init(model, gain=1.):
    def init_fn(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain)
            m.bias.data.fill_(0.01)
    model.apply(init_fn)


def train_2_model(model1, model2, n_epochs=50, lr=0.0001):
    post_model2 = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

    loss_fn = nn.MSELoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(list(model2.parameters()) + list(post_model2.parameters()), lr=lr)
    batch_size=32

    for epoch in range(n_epochs):
        # Generate random input
        npt = torch.rand(batch_size, 10) * 20 - 10

        # Target is the a Gaussian distribution around the input
        target = true_pre_model(npt)
        bottle1 = model1(npt)
        bottle2 = model2(npt)
        output2 = post_model2(bottle2)

        # Mask the bottleneck in mask_pcnt of the points
        mask_pcnt = 0.9
        loss_mask = torch.rand(batch_size) < mask_pcnt

        # Calculate loss
        loss1 = loss_fn(target=target[loss_mask], input=bottle1[loss_mask]) * (1 / (1 - mask_pcnt))
        loss2 = loss_fn(target=target[loss_mask], input=bottle2[loss_mask])

        loss2 += loss_fn(target=true_post_model(target), input=output2)

        # Backpropagate
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        # # Print loss every 100 epochs
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}: {loss.item()}")
        #     loss_inside = loss_fn(target=target[~loss_mask], input=y[~loss_mask])
        #     print(f"Loss inside circle: {loss_inside.item()}, Loss outside circle: {loss.item()}")
        #     print(f"ratio: {loss_inside.item() / loss.item()}")
    
    final_loss1 = 0.
    final_loss2 = 0.

    for test_epoch in range(10):
        npt = torch.rand(batch_size, 10) * 20 - 10
        target = true_pre_model(npt)
        output1 = model1(npt)
        output2 = model2(npt)

        loss1 = loss_fn(target=target, input=output1)
        loss2 = loss_fn(target=target, input=output2)

        final_loss1 += loss1.item()
        final_loss2 += loss2.item()

    final_ratio = final_loss2 / final_loss1
    print(f"Final ratio: {final_ratio}")
    return final_ratio

def train_model(model, n_epochs=1000, lr=0.0001):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size=32

    for epoch in range(n_epochs):
        # Generate random input
        npt = torch.rand(batch_size, 2) * 20 - 10

        # Target is the a Gaussian distribution around the input
        target = true_model(npt)
        output = model(npt)

        # Only give loss if input is more than 5 away from the point (0, 5)
        x = npt[:, 0]
        y = npt[:, 1]
        # dist = torch.sqrt(x ** 2 + (y - 5) ** 2)
        # loss_mask = dist > 5

        loss_mask = torch.logical_and(x>y, torch.logical_and(x>0, y>0))

        # Calculate loss
        loss = loss_fn(target=target[loss_mask], input=output[loss_mask])

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # Print loss every 100 epochs
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}: {loss.item()}")
        #     loss_inside = loss_fn(target=target[~loss_mask], input=y[~loss_mask])
        #     print(f"Loss inside circle: {loss_inside.item()}, Loss outside circle: {loss.item()}")
        #     print(f"ratio: {loss_inside.item() / loss.item()}")
    
    final_loss = 0.
    final_loss_inside = 0.

    for test_epoch in range(10):
        npt = torch.rand(batch_size, 2) * 20 - 10
        target = true_model(npt)
        output = model(npt)
        x = npt[:, 0]
        y = npt[:, 1]
        
        loss_mask = torch.logical_and(x>y, torch.logical_and(x>0, y>0))

        loss = loss_fn(target=target[loss_mask], input=output[loss_mask])
        loss_inside = loss_fn(target=target[~loss_mask], input=output[~loss_mask])

        if not torch.isnan(loss) and not torch.isnan(loss_inside):
            final_loss += loss.item()
            final_loss_inside += loss_inside.item()

    final_ratio = final_loss_inside / final_loss
    print(f"Final ratio: {final_ratio}")
    return final_ratio


# Graphing the output of pre_model on a 2D space
# by simulating a grid of points and mapping the output
# to a 2D colour space
def graph_output(model):
    # Create grid of points
    x = torch.linspace(-10, 10, 100)
    y = torch.linspace(-10, 10, 100)
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
    circle = plt.Circle((0, 5), radius=5, color="g", fill=False)
    axs[0].add_artist(circle)
    axs[0].imshow(output, extent=[-10, 10, -10, 10])

    # On  the right, plot the target function
    target = true_model(grid)
    target = target.view(100, 100, 1)
    target = (target - target.min()) / (target.max() - target.min())
    target = torch.cat((target, torch.zeros(100, 100, 2)), dim=2)
    target = target.detach().numpy()
    axs[1].imshow(target, extent=[-1, 1, -1, 1])
    plt.show()


if __name__ == "__main__":
    ratios = []
    for _ in range(100):
        model1 = BaseModel()
        model2 = BaseModel()
        meta_init(model1)
        meta_init(model2)

        meta_init(true_pre_model, gain=5)
        meta_init(true_post_model, gain=5)
        ratios.append(train_2_model(model1, model2))
    
    # Safe geometric mean using logs
    print(f"Geometric mean ratio: {torch.exp(torch.tensor(ratios).log().mean())}")
        
    # graph_output(pre_model)
