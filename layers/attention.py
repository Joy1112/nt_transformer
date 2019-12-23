from __future__ import division
import numpy as onp
import jax.numpy as np
import jax.random as random
import neural_tangents as nt
from jax.experimental import stax
from neural_tangents import stax as nt_stax


def maskAttentionWeight(w):
    n = w.shape[-1]
    b = np.reshape(np.tril(np.ones([n, n])), [1, 1, n, n])
    return w * b - 1e9 * (1 - b)


def _attention(qkv_bt_3h_r):
    """
    qkv_bt_3h_r:
        shape: [B, T, 3H, R]
    """
    B, T, H_3, R = qkv_bt_3h_r.shape

    def init_fun(rng, input_shape):
        output_shape = (B, T, int(H_3 / 3), R)
        return output_shape, ()
    
    def apply_fun(params, inputs, **kwargs):
        q_bthr, k_bthr, v_bthr = np.splist(qkv_bt_3h_r, 3, axis=2)
        q_bhtr = np.transpose(q_bthr, [0, 2, 1, 3])
        v_bhtr = np.transpose(v_bthr, [0, 2, 1, 3])
        k_bhrt = np.transpose(k_bthr, [0, 2, 3, 1])
        W_bhtt = np.matmul(q_bhtr, k_bhrt) / np.sqrt(R)
        W_bhtt = maskAttentionWeight(W_bhtt)
        W_bhtt = stax.softmax(W_bhtt, axis=-1)

        return np.matmul(W_bhtt, v_bhtr)

    return init_fun, apply_fun

def attention()
