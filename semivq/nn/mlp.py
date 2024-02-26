import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self._liner1 = nn.Linear(input_size, hidden_size)
        self._liner2 = nn.Linear(hidden_size, hidden_size)
        self._liner3 = nn.Linear(hidden_size, output_size)
        self._norm1 = nn.BatchNorm1d(hidden_size)
        self._norm2 = nn.BatchNorm1d(hidden_size)
        #self._norm3 = nn.BatchNorm1d(output_size)
        # self._liner4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self._liner1(x)
        x = self._norm1(x)
        x = F.elu(x)
        x = self._liner2(x)
        x = self._norm2(x)
        x = F.elu(x)
        x = self._liner3(x)
        # x = self._norm3(x)
        # x = F.elu(x)
        return x
