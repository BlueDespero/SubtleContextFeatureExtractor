import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x