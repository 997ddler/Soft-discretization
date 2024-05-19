import torch
import numpy as np
import VQ_VAE


def compute(values=[]):
    mean = np.mean(values)
    std = np.std(values, ddof=1)

torch.cuda.set_device(3)
all_results = {}
for i in range(5):
    results = VQ_VAE.run_model(i)
    if i == 0:
        for key, value in results.items():
            all_results[key[ : -2]] = [value]
    else:
        for key, value in results.items():
            all_results[key[ : -2]].append(value)

for key, _ in all_results.items():
    rec_list = [] 
    lpips_list = []
    perplexity_list = []
    for result in all_results[key]:
        rec_list.append(result['rec loss'])
        lpips_list.append(result['lpips'])
        perplexity_list.append(result['perplexity'])
    print('-----------------' + key + '-----------------')
    print(f'rec loss mean: {np.mean(rec_list):.5f} | ' + \
          f'rec loss std: {np.std(rec_list):.5f}'
         )
    print(f'lpips loss mean: {np.mean(lpips_list):.5f} | ' + \
          f'lpips loss std: {np.std(lpips_list):.5f}'
         )
    print(f'perplexity mean: {np.mean(perplexity_list):.5f} | ' + \
          f'perplexity std: {np.std(perplexity_list):.5f}'
         )

