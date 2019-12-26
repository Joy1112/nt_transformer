from __future__ import division
import numpy as onp
import jax.numpy as np
import jax.random as random
import neural_tangents as nt
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from torchnlp.word_to_vector import FastText, GloVe
from jax.experimental import stax
from neural_tangents import stax as nt_stax
from lib.utils import getCor


if __name__ == "__main__":
    vectors = GloVe()

    sent1 = "The brown fox jumps over the dog".split()
    sent2 = "The quick brown fox jumps over the lazy dog".split()

    inputseqs = [
        list(map(lambda w: vectors[w].numpy(), sent1)),
        list(map(lambda w: vectors[w].numpy(), sent2))
    ]

    embedarr = np.array(inputseqs[0] + inputseqs[1])
    embedcov = embedarr @ embedarr.T / embedarr.shape[1]



    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    ax = plt.gca()
    plt.imshow(embedcov, cmap='PuBu_r')
    span = np.linspace(-.5, 15.5)
    plt.plot(span, [6.5] * len(span), 'r')
    plt.plot([6.5] * len(span), span, 'r')
    plt.yticks(np.arange(16), sent1 + sent2)
    plt.xticks(np.arange(16), sent1 + sent2, rotation=90)
    plt.title('GloVe covariances')
    plt.xlabel('sent1                       sent2')
    plt.ylabel('sent2                       sent1')
    plt.colorbar()
    plt.grid()

    plt.subplot(122)
    ax = plt.gca()
    plt.imshow(getCor(embedcov), cmap='viridis')
    span = np.linspace(-.5, 15.5)
    plt.plot(span, [6.5] * len(span), 'r')
    plt.plot([6.5] * len(span), span, 'r')
    plt.yticks(np.arange(16), sent1 + sent2)
    plt.xticks(np.arange(16), sent1 + sent2, rotation=90)
    plt.title('GloVe correlations')
    plt.xlabel('sent1                       sent2')
    plt.ylabel('sent2                       sent1')
    plt.colorbar()
    plt.grid()

    plt.tight_layout()
