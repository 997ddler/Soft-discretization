import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold

class my_plot():
    instance = None
    def __init__(self):
        self._fig = plt.figure(figsize=(24, 8))
        self._perplexity = self._fig.add_subplot(1, 3, 1)
        self._perplexity.set_yscale('log')
        self._perplexity.set_title('Smoothed codebook perplexity.')
        self._perplexity.set_xlabel('iteration')

        self._loss = self._fig.add_subplot(1, 3, 2)
        self._loss.set_yscale('log')
        self._loss.set_title('Smoothed NMSE.')
        self._loss.set_xlabel('iteration')

        # self._active = self._fig.add_subplot(1, 5, 3)
        # self._active.set_title('Active Ratio.')
        # self._active.set_xlabel('iteration')

        self._lpips = self._fig.add_subplot(1, 3, 3)
        self._lpips.set_yscale('log')
        self._lpips.set_title('LPIPS.')
        self._lpips.set_xlabel('iteration')

        self._bars = []

        self._handle = []
        self._labels = []
        return

    def update(self, perplexity, loss, active, lpips, alpha, model_name=None):
        if model_name is None:
            raise ValueError("model_name is None")
        self._handle.append(self._perplexity.plot(perplexity, label=model_name))
        self._loss.plot(loss)
        # self._active.plot(active)
        self._lpips.plot(lpips)
        self._labels.append(model_name)
        if alpha is not None:
            plt.figure(figsize=(8, 8))

            # median and mean
            median_alpha = np.median(alpha)
            mean_alpha = np.mean(alpha)
            variance_alpha = np.var(alpha)

            plt.axhline(y=median_alpha, color='g', linestyle='--', label=f'Median ({median_alpha:.2f})')
            plt.axhline(y=mean_alpha, color='y', linestyle='-', label=f'Mean ({mean_alpha:.2f})')

            # color
            color = ['b' if val >= 0.0 else 'r' for val in alpha]
            alpha = np.abs(alpha)
            plt.bar(x=np.arange(len(alpha)), height=np.abs(alpha), color=color)
            plt.plot([], [], ' ', label=f'Variance ({variance_alpha:.2f})')

            plt.xlabel('Index')
            plt.ylabel('Alpha Value')

            plt.legend(loc='upper right')
            plt.title(model_name)
            plt.bar(x=np.arange(0, len(alpha)), height=alpha, color=color)
            plt.savefig('D:/discrete representation/vqtorch-main/Experiments/alpha' + '(' + model_name + ')' + '.png')


    def plot_tSNE(self, vectors, model_name = None, last_mean = None):
        mean = np.mean(vectors, axis=0)
        end = -1
        np.concatenate([vectors, mean.reshape(1, -1)], axis=0)
        if last_mean is not None:
            np.concatenate([vectors, last_mean.detach().cpu().numpy().reshape(1, -1)], axis=0)
            end = -2
        tSNE = manifold.TSNE(n_components=2, init='pca', random_state=0)
        Y = tSNE.fit_transform(vectors)
        plt.figure(figsize=(16, 8))
        plt.title(model_name)
        if end == -1:
            plt.scatter(Y[:end, 0], Y[:end, 1], c='red')
            plt.scatter(Y[end, 0], Y[end, 1], c='blue')
        else:
            plt.scatter(Y[:end, 0], Y[:end, 1], c='red')
            plt.scatter(Y[end, 0], Y[end, 1], c='blue')
            plt.scatter(Y[end + 1, 0], Y[end + 1, 1], c='k')
        plt.savefig('D:/discrete representation/vqtorch-main/Experiments/tSNE' + model_name + '.png')

    def save_fig(self):
        self._fig.savefig('D:/discrete representation/vqtorch-main/Experiments/training_loss.png')

    def already(self):
        self._fig.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize='small')

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance