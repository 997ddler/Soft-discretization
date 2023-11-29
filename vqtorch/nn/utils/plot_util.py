import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold

class my_plot():
    instance = None
    def __init__(self):
        self._fig = plt.figure(figsize=(16, 8))
        self._perplexity = self._fig.add_subplot(1, 5, 1)
        self._perplexity.set_yscale('log')
        self._perplexity.set_title('Smoothed codebook perplexity.')
        self._perplexity.set_xlabel('iteration')

        self._loss = self._fig.add_subplot(1, 5, 2)
        self._loss.set_yscale('log')
        self._loss.set_title('Smoothed NMSE.')
        self._loss.set_xlabel('iteration')

        self._active = self._fig.add_subplot(1, 5, 3)
        self._active.set_title('Active Ratio.')
        self._active.set_xlabel('iteration')

        self._lpips = self._fig.add_subplot(1, 5, 4)
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
        self._active.plot(active)
        self._lpips.plot(lpips)
        self._labels.append(model_name)
        if alpha is not None:
            plt.figure(figsize=(16, 8))
            color = ['b' if val >= 0.0 else 'r' for val in alpha]
            alpha = np.abs(alpha)
            plt.title(model_name)
            plt.bar(x=np.arange(0, len(alpha)), height=alpha, color=color)


    def plot_tSNE(self, vectors, model_name = None):
        mean = np.mean(vectors, axis=0)
        np.concatenate([vectors, mean.reshape(1, -1)], axis=0)
        tSNE = manifold.TSNE(n_components=2, init='pca', random_state=0)
        Y = tSNE.fit_transform(vectors)
        plt.figure(figsize=(16, 8))
        plt.title(model_name)
        plt.scatter(Y[:-1, 0], Y[:-1, 1], c='red')
        plt.scatter(Y[-1, 0], Y[-1, 1], c='blue')
        plt.title("t-SNE")

    def already(self):
        self._fig.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize='small')

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance