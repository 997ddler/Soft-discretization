import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self._liner1 = nn.Linear(input_size, hidden_size)
        self._liner2 = nn.Linear(hidden_size, hidden_size)
        self._liner3 = nn.Linear(hidden_size, output_size)
        #C self._liner4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self._liner1(x)
        x = F.relu(x)
        x = self._liner2(x)
        x = F.relu(x)
        x = self._liner3(x)
        return x
