import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dims, hidden_dims)
        self.layer2 = torch.nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        return self.layer2(h)
