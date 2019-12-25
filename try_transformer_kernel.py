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

from layers.attention import attention
from model.transformer import encoderGroup


def main():
    LEARNING_RATE = 1.0
    TRAIN_SIZE = 128
    TEST_SIZE = TRAIN_SIZE
    TRAIN_TIME = 1000.0

    N_STATE = [2**i for i in range(5, 13)]
    # N_STATE = 2048
    N_HEAD = 1
    N_GROUP = 2
    D_INIT = 300

    # x_train, y_train, x_test, y_test = datasets.get_dataset('mnist', TRAIN_SIZE, TEST_SIZE)

    vectors = GloVe()
    sent1 = "Machine learning is very interesting to me".split()
    sent2 = "Machine learning is not very interesting to my brother".split()

    inputseqs = [list(map(lambda w: vectors[w].numpy(), sent1)), list(map(lambda w: vectors[w].numpy(), sent2))]
    
    for n_state in N_STATE:
        inputseqs_embedded = inputseqs * onp.random.randn(D_INIT, n_state) * onp.sqrt(1 / D_INIT)
        if len(inputseqs_embedded.shape) < 3:
            inputseqs_embedded = onp.expand_dims(inputseqs_embedded, axis=0)
        init_fn, apply_fn = encoderGroup(N_GROUP, n_state, N_HEAD)
        for i in range(10):
            key = random.PRNGKey(0)
            _, params = init_fn(key, inputseqs_embedded.shape)
            outvecs1 = apply_fn(params, inputseqs_embedded[:, :7])
            outvecs2 = apply_fn(params, inputseqs_embedded[:, 7:])
            outvecs = np.concatenate([outvecs1[0], outvecs2[0]])
            outgram = outvecs * outvecs.T / outvecs.shape[1]
            print(outgram)
            break

if __name__ == "__main__":
    main()
