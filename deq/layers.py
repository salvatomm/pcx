"""Reusable layers for DEQ-PC models."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import pcx as px
import pcx.nn as pxnn


class GroupNorm(px.Module):
    """Channel-first GroupNorm (no running stats)."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = px.static(num_groups)
        self.eps = px.static(eps)
        self.weight = pxnn.LayerParam(jnp.ones(num_channels))
        self.bias = pxnn.LayerParam(jnp.zeros(num_channels))

    def __call__(self, x: jax.Array) -> jax.Array:
        c = x.shape[0]
        g = self.num_groups.get()
        xg = x.reshape(g, c // g, *x.shape[1:])
        axes = tuple(range(1, xg.ndim))
        mean = jnp.mean(xg, axis=axes, keepdims=True)
        var = jnp.var(xg, axis=axes, keepdims=True)
        xg = (xg - mean) / jnp.sqrt(var + self.eps.get())
        xg = xg.reshape(x.shape)
        shape = (c,) + (1,) * (x.ndim - 1)
        return xg * self.weight.get().reshape(shape) + self.bias.get().reshape(shape)


def _small_init_conv(conv, init_scale: float):
    """Re-initialise all LayerParams in a conv with N(0, init_scale)."""
    leaves = jtu.tree_leaves(conv, is_leaf=lambda x: isinstance(x, pxnn.LayerParam))
    for leaf in leaves:
        if isinstance(leaf, pxnn.LayerParam):
            leaf.set(jax.random.normal(px.RKG(), leaf.shape) * init_scale)


class ConvBlock(px.Module):
    """Single conv + residual + input injection + GroupNorm.

    f(z, x_inj) = GroupNorm(ReLU(z + Conv(z) + x_inj))
    """

    def __init__(self, n_ch: int, ks: int = 3, ng: int = 8, init_scale: float = 0.005):
        super().__init__()
        p = ks // 2
        self.conv = pxnn.Conv2d(n_ch, n_ch, ks, padding=(p, p), use_bias=False)
        self.norm = GroupNorm(ng, n_ch)
        self.init_scale = px.static(init_scale)
        _small_init_conv(self.conv, init_scale)

    def __call__(self, z: jax.Array, x_inj: jax.Array) -> jax.Array:
        return self.norm(jax.nn.relu(z + self.conv(z) + x_inj))


class ResNetLayer(px.Module):
    """Original 2-conv residual block (for single-vode model)."""

    def __init__(self, n_ch: int, n_inner: int, ks: int = 3, ng: int = 8,
                 init_scale: float = 0.001):
        super().__init__()
        p = ks // 2
        self.conv1 = pxnn.Conv2d(n_ch, n_inner, ks, padding=(p, p), use_bias=False)
        self.conv2 = pxnn.Conv2d(n_inner, n_ch, ks, padding=(p, p), use_bias=False)
        self.norm1 = GroupNorm(ng, n_inner)
        self.norm2 = GroupNorm(ng, n_ch)
        self.norm3 = GroupNorm(ng, n_ch)
        self.init_scale = px.static(init_scale)
        _small_init_conv(self.conv1, init_scale)
        _small_init_conv(self.conv2, init_scale)

    def __call__(self, z: jax.Array, x: jax.Array) -> jax.Array:
        y = self.norm1(jax.nn.relu(self.conv1(z)))
        return self.norm3(jax.nn.relu(z + self.norm2(x + self.conv2(y))))
    

def _small_init_linear(linear, init_scale: float):
    """Re-initialise all LayerParams in a pxnn.Linear with N(0, init_scale)."""
    leaves = jtu.tree_leaves(linear, is_leaf=lambda x: isinstance(x, pxnn.LayerParam))
    for leaf in leaves:
        if isinstance(leaf, pxnn.LayerParam):
            leaf.set(jax.random.normal(px.RKG(), leaf.shape) * init_scale)


class FCBlock(px.Module):
    """Fully-connected implicit block with residual + input injection + LayerNorm.

    f(z, x_inj) = LayerNorm(ReLU(z + Linear(z) + x_inj))

    Mirrors ConvBlock but replaces Conv2d + GroupNorm with Linear + LayerNorm.
    """

    def __init__(self, dim: int, init_scale: float = 0.005):
        super().__init__()
        self.linear = pxnn.Linear(dim, dim, bias=False)
        self.norm = pxnn.LayerNorm((dim,))
        _small_init_linear(self.linear, init_scale)

    def __call__(self, z: jax.Array, x_inj: jax.Array) -> jax.Array:
        return self.norm(jax.nn.relu(z + self.linear(z) + x_inj))