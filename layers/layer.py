from __future__ import division
import numpy as onp
import jax.numpy as np
import jax.random as random
import neural_tangents as nt
from jax.experimental import stax
from neural_tangents import stax as nt_stax


def randn(shape, std):
    return onp.random.randn(*shape).astype(np.float32) * std


def _layerNorm(x, axis, g=None, b=None, e=1e-5):
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.mean(np.square(x - mu), axis=axis, keepdims=True)
    sigma = np.sqrt(sigma + e)
    x = (x - mu) / sigma
    if g is not None and b is not None:
        x = x * g + b
    return x


def layerNorm(cx, x, axis=-1):
    n_state = x.shape[axis]
    g = cx.get_variable("g", initializer=lambda: onp.ones(n_state, 'f'))
    b = cx.get_variable("b", initializer=lambda: onp.zeros(n_state, "f"))

    return _layerNorm(x, axis=axis, g=g, b=b)


def normax(shape, axis):
    out = onp.random.randn(*shape).astype(np.float32)
    return out / onp.sqrt(onp.square(out).sum(axis, keepdims=True))


def normc(*shape):
    return normax(shape, axis=0)


def dense(cx, x, F):
    # batch b, sequences t, d_model k
    B, T, K = x.shape
    x_bt_k = np.reshape(x, [-1, K])
    W_k_f = cx.get_variable("W", initializer=lambda: normc(K, F))
    b_f = cx.get_variable("b", initializer=lambda: onp.zeros(F, 'f'))
    y_bt_f = np.matmul(x_bt_k, W_k_f) + b_f

    return np.reshape(y_bt_f, [B, T, F])


def mlp(cx, x, n_hidden):
    S = x.shape[-1]
    H_b_t_h = stax.relu(dense(cx.scope('c_fc'), x, n_hidden))
    Y_b_t_s = dense(cx.scope('c_proj'), H_b_t_h, S)

    return Y_b_t_s
