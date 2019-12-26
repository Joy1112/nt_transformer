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


def dense_old(cx, x, F):
    # batch b, sequences t, d_model k
    B, T, K = x.shape
    x_bt_k = np.reshape(x, [-1, K])
    W_k_f = cx.get_variable("W", initializer=lambda: normc(K, F))
    b_f = cx.get_variable("b", initializer=lambda: onp.zeros(F, 'f'))
    y_bt_f = np.matmul(x_bt_k, W_k_f) + b_f

    return np.reshape(y_bt_f, [B, T, F])


def mlp_old(cx, x, n_hidden):
    S = x.shape[-1]
    H_b_t_h = stax.relu(dense(cx.scope('c_fc'), x, n_hidden))
    Y_b_t_s = dense(cx.scope('c_proj'), H_b_t_h, S)

    return Y_b_t_s

def dense(cx, x, F):
    


def maskAttentionWeight(w):
    n = w.shape[-1]
    b = np.reshape(np.tril(np.ones([n, n])), [1, 1, n, n])
    return w * b - 1e9 * (1 - b)


def _Attention(q, k, v):
    """
    q:
        shape: [B, H, T, R]
    k:
        shape: [B, H, R, T]
    v:
        shape: [B, H, T, R]
    """

    R = q.shape[-1]
    W_bhtt = np.matmul(q, k) / np.sqrt(R)
    W_bhtt = maskAttentionWeight(W_bhtt)
    W_bhtt = stax.softmax(W_bhtt, axis=-1)

    return np.matmul(W_bhtt, v)


# def Attention(cx, x, n_state, n_head):
#     B, T, _K = x.shape
#     assert n_state % n_head == 0

    
