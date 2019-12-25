from __future__ import division
import numpy as onp
import jax.numpy as np
import jax.random as random
import neural_tangents as nt
from jax.experimental import stax
from neural_tangents import stax as nt_stax

from layers.attention import attention


def encoderBlock(n_state, n_head=1):
    Main = stax.serial(attention(n_state, n_head), stax.BatchNorm(axis=-1), stax.Dense(4 * n_state), stax.Relu(),
                       stax.Dense(n_state))
    Shortcut = stax.Identity()
    return stax.serial(stax.FanOut(2), stax.Parallel(Main, Shortcut), stax.FanInSum(), stax.BatchNorm(axis=-1))


def encoderGroup(n_group, n_state, n_head=1):
    blocks = []
    for _ in range(n_group):
        blocks += [encoderBlock(n_state, n_head)]
