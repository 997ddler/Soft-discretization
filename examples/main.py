import torch.cuda

import VQ_VAE

for i in range(2):
    VQ_VAE.run_model(i)