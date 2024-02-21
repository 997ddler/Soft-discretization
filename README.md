# Soft-discretization
This is a project based on discretization method to make VQ-VAE model have a better performance. We modify the source code of vqtorch to realize learnable standard deviation(In the code we name it alpha). You can browse the entire project starting from `examples\VQ_VAE.py`. If you want to use these methods, you can download the whole project from Git Hub. The `semivq` folder contains the encapsulated tools. You could import `semivq.nn.vq` as part of your module to implement quantization.



# Parameter illustration

`VectorQuant` contains a few parameters which controls its behavior.

```
	"""
	Vector quantization layer using straight-through estimation.

	Args:
		feature_size (int): feature dimension corresponding to the vectors
		num_codes (int): number of vectors in the codebook
		beta (float): commitment loss weighting
		sync_nu (float): sync loss weighting
		affine_lr (float): learning rate for affine transform (VQ-STE++ affine parameter learnable)
		affine_groups (int): number of affine parameter groups (VQ-STE++ affine parameter learnable)
		replace_freq (int): frequency to replace dead codes 
		inplace_optimizer (Optimizer): optimizer for inplace codebook updates (VQ-STE++ synchronized update)
		using_statistics: (bool) using EMA  to affine transform(VQ-STE++ affine parameter update the whole codebook using EMA)
		use_learnable_std: (bool) = False,  (our method Approach 1)
		alter_penalty : (str) = 'default',  (penalization alpha)
		use_learnable_gamma (bool) False,  (our method Approach 2&3)
		gamma_policy(bool) = 'default',   (select Approach 2&3)
		**kwargs: additional arguments for _VQBaseLayer
	
	Returns:
		Quantized vector z_q
        and
        return dict (			
			'z'  : z,               # each group input z_e
			'z_q': z_q,             # quantized output z_q
			'd'  : d,               # distance function for each group
			'q'	 : q,				# codes
			'loss': self.compute_loss(z, z_q).mean(), # commitment loss
			'perplexity': perplexity, # perplexity
			'active_ratio': active_ratio, # active ratio
			)
```



# Example 

```python
from semivq.nn.vq import VectorQuant
 
class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, feature_size, **vq_kwargs):    
        super().__init__()
        self.layers = nn.ModuleList([
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                VectorQuant(feature_size, **vq_kwargs),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
                ])
        return
    
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, VectorQuant):
                x, vq_dict = layer(x)
            else:
                x = layer(x)
        return x.clamp(-1, 1), vq_dict
  

def train(model, train_loader, train_iterations=1000, alpha=10):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.cuda(), y.cuda()
    
    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, vq_out = model(x)
        rec_loss = (out - x).abs().mean()
        cmt_loss = vq_out['loss']
        (rec_loss + alpha * cmt_loss).backward()

        opt.step()
        pbar.set_description(f'rec loss: {rec_loss.item():.3f} | ' + \
                             f'cmt loss: {cmt_loss.item():.3f} | ' + \
                             f'active %: {vq_out["q"].unique().numel() / num_codes * 100:.3f}')
    return

# Original VQ-VAE
model = SimpleVQAutoEncoder(feature_size)

# Our mehod Approach 1
model = SimpleVQAutoEncoder(feature_size, use_learnable_std=True)

# Our method Approach 2
model = SimpleVQAutoEncoder(feature_size, use_learnable_gamma=True)

# Our method Approach 3
model = SimpleVQAutoEncoder(feature_size, use_learnable_gamma=True, gamma_policy="personalize")
```

This is a simple example which shows that how to use the module `VectorQuant`
