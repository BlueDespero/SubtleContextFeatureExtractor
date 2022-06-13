import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.fc1 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x