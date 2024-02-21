import torch
from torch import nn
from semivq.nn.mlp import MLP


class gamma_learner(nn.Module):
    def __init__(self, feature_size, num_codes=1024, type='default'):
        super(gamma_learner, self).__init__()
        self._original_size = []
        self._gamma = None
        self.type = type
        if type == 'default':
            self._layer = MLP(num_codes, 1, feature_size)
        elif type == 'personalize':
            # self._layer = nn.Conv2d(
            #     in_channels=feature_size,
            #     out_channels=feature_size,
            #     kernel_size=(3, 1),
            #     stride=1,
            #     padding=0,
            #     groups=feature_size
            # )
            self._layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=feature_size ,
                    out_channels=feature_size,
                    kernel_size=(2, 1),
                    stride=1,
                    padding=0,
                    groups=feature_size
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=(2, 1),
                    stride=1,
                    padding=0,
                    groups=feature_size
                ),
            )

    def forward(self, z_q, codebook):
        if self.type == 'default':
            self._gamma = self._layer(codebook.T).T
            return self._gamma + z_q  # z_d
        elif self.type == 'personalize':
            # reshape data (B, H, W, C) => (B, C i.e feature_size, 1, H * W)
            z_q = self._reshpe_input(z_q)

            # compute codebook mean, var
            c_mean = codebook.mean([0]).unsqueeze(0)
            c_var = codebook.var([0], unbiased=False).unsqueeze(0)
            c_mean = c_mean.repeat(z_q.shape[0], z_q.shape[1], 1, 1)
            c_var = c_var.repeat(z_q.shape[0], z_q.shape[1], 1, 1)

            # forward
            input = torch.cat((c_mean, z_q, c_var), 2)
            self._gamma = self._layer(input)

            # get results
            z_d = self._resore_input(z_q) + self._resore_input(self._gamma)
            return z_d

    def _reshpe_input(self, x):
        self._original_size = (x.shape[1], x.shape[2])
        x = x.flatten(1, 2)
        x = x.permute(0, 3, 2, 1)
        return x

    def _resore_input(self, x):
        x = x.permute(0, 3, 2, 1)
        x = x.unflatten(1, self._original_size)
        return x