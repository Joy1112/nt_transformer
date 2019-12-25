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


def _attention(n_state, n_head=1):
    """
    qkv_bt_3h_r:
        shape: [B, T, 3H, R]
    """

    def init_fun(rng, input_shape):
        output_shape = (input_shape[0], input_shape[1], n_head, n_state // n_head)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        R = n_state // n_head
        qkv_bt_3h_r = np.reshape(inputs, [B, T, 3 * n_head, R])
        q_bthr, k_bthr, v_bthr = np.splist(qkv_bt_3h_r, 3, axis=2)
        q_bhtr = np.transpose(q_bthr, [0, 2, 1, 3])
        v_bhtr = np.transpose(v_bthr, [0, 2, 1, 3])
        k_bhrt = np.transpose(k_bthr, [0, 2, 3, 1])
        W_bhtt = np.matmul(q_bhtr, k_bhrt) / np.sqrt(R)
        W_bhtt = maskAttentionWeight(W_bhtt)
        W_bhtt = stax.softmax(W_bhtt, axis=-1)
        A_bhtr = np.matmul(W_bhtt, v_bhtr)
        A_bthr = np.transpose(A_bhtr, [0, 2, 1, 3])
        A_bts = np.reshape(A_bthr, [B, T, n_state])
        return A_bts

    return init_fun, apply_fun


def attention(n_state, n_head=1, residual=True):
    Main = stax.serial(stax.Dense(3 * n_state),
                       _attention(n_state, n_head),
                       stax.Dense(n_state))
    Shortcut = stax.Identity()
    return stax.serial(stax.FanOut(2), stax.Parallel(Main, Shortcut), stax.FanInSum())
