import torch

# so we're gonna randomly initialize some parameters

# We're going to have a lil baby version of the setup
# Parameters are the 
# so we have a true weight matrix, which it learns, and then it also has a downstream task
# which is represented by.. could be the ability to scale but not rotate the attr_dim

INPUT_DIM = 20
ATTR_DIM = 5
L1_LOSS_WEIGHT = 0.01

true_attr_weights = torch.eye(INPUT_DIM, ATTR_DIM)
true_attr_lin = torch.nn.Linear(INPUT_DIM, ATTR_DIM)
true_attr_lin.weight = torch.nn.Parameter(true_attr_weights.T)


weights = torch.nn.Linear(INPUT_DIM, ATTR_DIM)


output_feature = torch.rand(INPUT_DIM)

post_weight_vec = torch.nn.Linear(ATTR_DIM, 1)
optimizer = torch.optim.Adam(list(weights.parameters()) + list(post_weight_vec.parameters()), lr=0.01)

# Training loop
for ndx in range(1000):
    attr_weight = 1
    input_vec = torch.rand(INPUT_DIM)
    attr_vec = weights(input_vec)
    true_vec = true_attr_lin(input_vec).detach()

    downstream_vec = post_weight_vec(attr_vec)
    # torch for dot product of two vectors is
    target_downstream = torch.dot(output_feature, input_vec)
    downstream_loss = ((downstream_vec - target_downstream) ** 2).mean()

    attr_loss = ((attr_vec - true_vec) ** 2).mean() * attr_weight
    loss = downstream_loss + attr_loss + sum([L1_LOSS_WEIGHT * torch.norm(p) for p in post_weight_vec.parameters()])
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()



def entropy_test() -> None:
    # Going to work with binary functions defined by hyperplanes on the input space
    INPUT_DIM = 10

    # Defining n random hyperplanes, with vector components sampled randomly from [-1, 1]
    n = 1000
    hyperplane_directions = torch.rand(n, INPUT_DIM) - 0.5
    hyperplane_directions /= torch.norm(hyperplane_directions, dim=1, keepdim=True)
    hyperplane_biases = torch.rand(n) * 1 - 0.5

    # Calculate the entropy of the hyperplanes
    # This is the probability of a random point being in the positive halfspace
    # of each hyperplane

    # So we're going to sample a bunch of points
    n_samples = 100000
    samples = torch.rand(n_samples, INPUT_DIM) - 0.5
    
    # Now we're going to calculate the probability of each sample being in the positive halfspace

    # So we're going to calculate the dot product of each sample with each hyperplane
    # This will give us a matrix of size n_samples x n

    true_false_matrix = torch.matmul(samples, hyperplane_directions.T) > hyperplane_biases
    prop_true = true_false_matrix.float().mean(dim=0)
    # Apply entropy function, where entropy is zero if prop_true is 0 or 1
    entropy = -prop_true * torch.log(prop_true) - (1 - prop_true) * torch.log(1 - prop_true)
    entropy[torch.isnan(entropy)] = 0
    sum_true = true_false_matrix.int().sum(dim = 0)
    val, ind = torch.sort(sum_true)
    print(val)
    breakpoint()

if __name__ == "__main__":
    entropy_test()
