import math
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from vqtorch.nn.utils.TinyImagenet import TinyImageNet
from vqtorch.nn import VectorQuant
from vqtorch.nn.resnet import EncoderVqResnet32, DecoderVqResnet32
from torch.nn import functional as F
from vqtorch.nn.utils.plot_util import my_plot
import lpips as lpips
from vqtorch.nn.sq_vae import GaussianSQVAE, SQVAE
import warnings
warnings.filterwarnings("ignore")


class VQ_VAE(nn.Module):
    def __init__(
                self,
                dim_z=64, # feature dimension
                num_rb=2,
                flg_bn=True,
                flg_var_q=False,
                using_penalization=False,
                inner_learning_rate = 0.0,
                **vq_kwargs
    ):
        super().__init__()
        self._encoder = EncoderVqResnet32(dim_z, num_rb, flg_bn, flg_var_q)
        self._vq = VectorQuant(dim_z, **vq_kwargs)
        self._decoder = DecoderVqResnet32(dim_z, num_rb, flg_bn)
        self.using_penalization = using_penalization
        self.inner_learning_rate = inner_learning_rate
        return

    def forward(self, x):
        out = self._encoder(x)
        out, vq_dict = self._vq(out)
        out = self._decoder(out)
        return out, vq_dict, None

    def get_alpha(self):
        return self._vq.get_alpha()

    def getcodebook(self):
        return self._vq.get_codebook()

    def alpha_loss(self):
        return self._vq.alpha_loss()

    def get_dynamic_info(self):
        return self._vq.get_dynamic_info()

    def get_last_mean(self):
        return self._vq.get_last_mean()

    def is_schedule_learning_rate(self):
        return self.inner_learning_rate != 0.0

    def get_inner_learning_rate(self):
        return self.inner_learning_rate

    def get_inner_layer(self):
        return self._vq.get_inner_layer()


def train_SQVAE(model, train_loader, optimizer, epochs, cfgs, scheduler):
        rec_losses = []; perplexities = [];active_ratios = []; lpips_losses = []
        lpips_model = lpips.LPIPS(net='alex').cuda()
        model.train()
        i = 0
        for epoch in range(epochs):
            for batch_idx, (x, _) in enumerate(train_loader):
                if cfgs["decay"]:
                    step = (epoch - 1) * len(train_loader) + batch_idx + 1
                    temperature_current = np.max(
                        [cfgs["temperature_init"] * np.exp(-cfgs["temperature_decay"] * step),
                         cfgs["temperature_min"]]
                    )
                    model.quantizer.set_temperature(temperature_current)
                # input
                x = x.cuda()

                # forward
                out, vq_out, _ = model(x, flg_train=True, flg_quant_det=False)
                lpips_loss = lpips_model(x, out).mean()
                loss = lpips_loss + vq_out['all']

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # record
                rec_losses.append(vq_out["rec_loss"].cpu().item())
                perplexities.append(vq_out["perplexity"].detach().cpu().item())
                lpips_losses.append(lpips_loss.item())

                i += 1
                if i % 100 == 0:
                    print(f'rec loss: {np.mean(rec_losses[-100:]):.5f} | ' + \
                          f'perplexity %: {np.mean(perplexities[-100:]):.5f} | ' + \
                          f'lpips %: {np.mean(lpips_losses[-100:]):.5f}'
                          )

        # smooth
        train_res_recon_error_smooth = savgol_filter(rec_losses, 201, 7)
        train_res_perplexity_smooth = savgol_filter(perplexities, 201, 7)
        lpips_losses = savgol_filter(lpips_losses, 201, 7)
        my_plot.get_instance().update(train_res_perplexity_smooth, train_res_recon_error_smooth, lpips_losses,"SQ-VAE")


def train(model, model_name, optimizer, scheduler = None, train_loader = None, alpha=10, epochs = 15):
    i = 0
    rec_losses = []; perplexities = []; lpips_losses = []
    lpips_model = lpips.LPIPS(net='alex').cuda()
    model.train()
    for epoch in range(epochs):
        for data in train_loader:
            # get data
            image, _ = data
            image = image.cuda()
            optimizer.zero_grad()

            # forward
            out, vq_out, _ = model(image)
            lpips_loss = lpips_model(image, out).mean()
            rec_loss = F.mse_loss(out, image)
            cmt_loss = vq_out['loss']
            loss = rec_loss + alpha * cmt_loss + lpips_loss
            if model.using_penalization:
                loss += model.alpha_loss()

            # backward
            loss.backward()
            optimizer.step()

            scheduler.step()
            i += 1

            # record
            lpips_losses.append(lpips_loss.item()) 
            rec_losses.append(rec_loss.item())
            perplexities.append(vq_out["perplexity"].cpu().numpy())
            if i % 100 == 0:
                print(f'rec loss: {np.mean(rec_losses[-100 : ]):.5f} | ' + \
                      f'perplexity %: {np.mean(perplexities[-100 : ]):.5f} | ' + \
                      f'lpips %: {np.mean(lpips_losses[-100 : ]):.5f}'
                      )

        print(model_name + f' epoch {epoch + 1} finished ')
    # smooth
    train_res_recon_error_smooth = savgol_filter(rec_losses, 201, 7)
    train_res_perplexity_smooth = savgol_filter(perplexities, 201, 7)
    lpips_losses = savgol_filter(lpips_losses, 201, 7)

    alpha_info = model.get_dynamic_info()
    if alpha_info is not None:
        alpha0 = savgol_filter(alpha_info[0], 201, 7)
        alpha1 = savgol_filter(alpha_info[1], 201, 7)
        alpha_mean = savgol_filter(alpha_info[2], 201, 7)

    # plot
    my_plot.get_instance().update(
        train_res_perplexity_smooth,
        train_res_recon_error_smooth,
        lpips_losses,
        model_name
    )

    last_mean = model.get_last_mean()
    if alpha_info is not None:
        alpha_learnable = np.array(torch.squeeze(model.get_alpha()).detach().cpu().numpy())
        my_plot.get_instance().plot_change_alpha([alpha0, alpha1, alpha_mean], model_name)
        my_plot.get_instance().plot_alpha(alpha_learnable, model_name)

    my_plot.get_instance().plot_tSNE(model.getcodebook().detach().cpu().numpy(), model_name, last_mean)


def test(model, test_loader, type = 'test'):
    def show_image(original, generate, type = 'test'):
            npimg = original.numpy()
            plt.figure()
            plt.suptitle(type)

            plt.subplot(1, 2, 1)
            fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            npimg = generate.numpy()
            plt.subplot(1, 2, 2)
            fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

    print('\n ------------------------------Test begin ------------------------------\n')

    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    model.eval()
    rec_losses = []; lpips_losses = []; perplexities = []; active_ratios = []
    with torch.no_grad():
        for data in test_loader:
            image, _ = data
            image = image.cuda()
            if isinstance(model, SQVAE):
                out, vq_out, _ = model(image, flg_train=False, flg_quant_det=True)
                rec_loss = vq_out['rec_loss']
            else:
                out, vq_out, _ = model(image)
                rec_loss = F.mse_loss(out, image)

            d = loss_fn_alex(image, out).mean()
            lpips_losses.append(d.cpu().numpy())
            rec_losses.append(rec_loss.item())
            perplexities.append(vq_out["perplexity"].cpu().numpy())

    print(f'rec loss: {np.mean(rec_losses):.5f} | ' + \
            f'perplexity : {np.mean(perplexities):.5f} | ')

    test_dict = {
        'rec loss': np.mean(rec_losses),
        'perplexity': np.mean(perplexities),
        'active %': np.mean(active_ratios),
        'lpips': np.mean(lpips_losses)
    }

    image, _= next(iter(test_loader))
    image = image.cuda()
    out, _, _= model(image)

    show_image(make_grid(image.cpu().data), make_grid(out.cpu().data), type)
    return test_dict


def get_cosine_scheduler(optimizer, max_lr, min_lr, warmup_iter, base_lr, T_max):
    rule = lambda cur_iter : 1.0 if warmup_iter < cur_iter else (min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos((cur_iter - warmup_iter) / (T_max-warmup_iter) * math.pi))) / base_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)
    return scheduler


def merge_datasets(dataset, sub_dataset):
    # merge classes
    dataset.classes.extend(sub_dataset.classes)
    dataset.classes = sorted(list(set(dataset.classes)))
    # merge class_to_idx
    dataset.class_to_idx.update(sub_dataset.class_to_idx)
    # merge samples
    dataset.samples.extend(sub_dataset.samples)
    # merge targets
    dataset.targets.extend(sub_dataset.targets)


def load_dataset(batch_size = 256):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])

    train_loader = DataLoader(datasets.CIFAR10(root='~/data/cifar', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10(root='~/data/cifar', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=True)

    # path = "D:/discrete representation/vqtorch-main/img_align_celeba"
    #
    # data = ImageFolder(root=path, transform=transform)
    # train_data, test_data =torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(0.8 * len(data))])
    #
    # train_loader = DataLoader(train_data,
    #                                    batch_size=batch_size,
    #                                    shuffle=True,
    #                                    pin_memory=True
    #                           )
    #
    # test_loader = DataLoader(test_data,
    #                         batch_size=batch_size,
    #                         shuffle=True,
    #                         pin_memory=True)
    #
    # path = "D:/discrete representation/vqtorch-main/tiny-imagenet-200"
    # train_loader = DataLoader(TinyImageNet(path, True, transform), batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(TinyImageNet(path, False, transform), batch_size=batch_size, shuffle=True)


    return train_loader, test_loader

def run_model(times):
    print('\n------------------------------Start ------------------------------\n')
    seed = random.randint(0, 1000000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # hyperparameters
    batch_size = 128
    train_loader, test_loader = load_dataset(batch_size)
    num_codes = 1024
    learning_rate = 1e-4
    epochs = 1
    alpha = 5
    warm_epochs = 0
    weight_decay = 1e-4
    dict_dim = 64

    config = {
        "dataset_space" : 32 * 32 * 3,
        "param_var_q" : "gaussian_1",
        "dict_size" : num_codes,
        "dict_dim" : dict_dim,
        "log_param_q_init" : math.log(20),
        "temperature_decay" : 1e-5,
        "temperature_init" : 1.0,
        "temperature_min" : 0,
        "arelbo" : True,
        "decay" : True,
        "flg_var_q" : False,
        "flg_bn" : True,
    }

    inplace_optimizer1 = lambda *args, **kwargs: torch.optim.AdamW(*args, **kwargs, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
    inplace_optimizer2 = lambda *args, **kwargs: torch.optim.AdamW(*args, **kwargs, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
    dict = {
            #"VQ_VAE" : VQ_VAE(num_codes=num_codes).cuda(),
            #"VQ_STE++(learnable)" : VQ_VAE(num_codes=num_codes, sync_nu=2.0, affine_lr=2.0, dim_z = dict_dim, beta=1.0, inplace_optimizer = inplace_optimizer1).cuda(),
            #"VQ_STE++(statistical)" : VQ_VAE(num_codes=num_codes, sync_nu=2.0, affine_lr=2.0, using_statistics=True, dim_z = dict_dim, beta=1.0, inplace_optimizer = inplace_optimizer2).cuda(),
            #"SQ-VAE": GaussianSQVAE(config).cuda(),
            "VQ_VAE + learn. alpha" : VQ_VAE(num_codes=num_codes, use_learnable_std=True, dim_z=dict_dim).cuda(),
            "VQ_VAE + learn. alpha + schedule lr" : VQ_VAE(num_codes=num_codes, use_learnable_std=True, use_learnable_mean=False, dim_z=dict_dim, inner_learning_rate=learning_rate * 10).cuda(),
            "VQ_VAE + learnable std + penalization -(alpha)^2" : VQ_VAE(num_codes=num_codes, use_learnable_std=True, dim_z=dict_dim, using_penalization=True).cuda(),
            "VQ_VAE + learnable std + penalization (1-alpha)^2" : VQ_VAE(num_codes=num_codes, use_learnable_std=True, dim_z=dict_dim, using_penalization=True, alter_penalty="between1").cuda(),
    }

    result = {}
    for key, value in dict.items():
        if not isinstance(value, SQVAE) and value.is_schedule_learning_rate():
            base_params = filter(lambda p: id(p) not in map(id, value.get_inner_layer().parameters()), value.parameters())
            optimizer = torch.optim.AdamW(
                params=[
                    {"params" : base_params},
                    {"params" : value.get_inner_layer().parameters(), "lr":value.get_inner_learning_rate()}
                ],
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95)
            )
        else:
            optimizer = torch.optim.AdamW(value.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
        scheduler = get_cosine_scheduler(optimizer, learning_rate, 1e-5, warm_epochs * len(train_loader), learning_rate, epochs * len(train_loader))
        key = key + '  ' + str(times)
        if isinstance(value, SQVAE):
            train_SQVAE(value, train_loader, optimizer, epochs, config, scheduler)
            result[key] = test(value, test_loader, key)
        else:
            train(value, key, optimizer, scheduler, train_loader, alpha=alpha, epochs=epochs)
            result[key] = test(value, test_loader, key)
        print('\n-------------------------------Test End-----------------------------\n')


    print('\n------------------------------Result ------------------------------\n')
    for key, value in result.items():
        print('\n-----' + key + '-----\n')
        print(f'rec loss: {value["rec loss"]:.5f} | ' + \
                f'perplexity : {value["perplexity"]:.5f} | ' + \
                f'active %: {value["active %"]:.5f} | ' + \
                f'lpips : {value["lpips"]:.5f}')

    my_plot.get_instance().already()
    my_plot.get_instance().save_fig()
    plt.show()
    print('seed:' + str(seed))
    print('\n------------------------------End ------------------------------\n')