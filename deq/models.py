"""DEQ-PC model definitions.

Both models accept ``energy_type: str`` (``"se"`` or ``"log_cosh"``).
This controls:
  1. The energy function inside each latent Vode (used by ``model.energy()``
     when computing weight gradients).
  2. The diff energy used in the manual inference loop (read via
     ``model.energy_type.get()`` — a ``px.static``, so JIT resolves the
     branch at trace time with zero runtime cost).
"""

import jax
import jax.numpy as jnp

import pcx as px
import pcx.nn as pxnn
import pcx.predictive_coding as pxc

from .layers import GroupNorm, ResNetLayer, ConvBlock
from .energies import nudged_ce_energy, get_vode_energy


# ======================================================================
# Single-Vode DEQ  (original architecture)
# ======================================================================

class DEQPCModel(pxc.EnergyModule):
    """Single implicit layer: z* = f(z*, x_inj)."""

    def __init__(
        self,
        n_channels: int = 48,
        n_inner: int = 64,
        n_classes: int = 10,
        nudging: float = 0.01,
        init_scale: float = 0.001,
        stop_grad_f: bool = False,
        energy_type: str = "se",
    ):
        super().__init__()
        self.n_classes = px.static(n_classes)
        self.n_channels = px.static(n_channels)
        self.nudging = px.static(nudging)
        self.init_scale = px.static(init_scale)
        self.stop_grad_f = px.static(stop_grad_f)
        self.energy_type = px.static(energy_type)

        # Encoder
        self.input_conv = pxnn.Conv2d(3, n_channels, 3, padding=(1, 1), use_bias=True)
        self.gn_in = GroupNorm(8, n_channels)

        # Implicit block
        self.f = ResNetLayer(n_channels, n_inner, init_scale=init_scale)

        # Readout
        self.gn_out = GroupNorm(8, n_channels)
        self.pool = pxnn.AvgPool2d(kernel_size=8, stride=8)
        self.linear = pxnn.Linear(n_channels * 4 * 4, n_classes)

        # Vodes — latent uses energy_type, output uses nudged CE
        self.vode_z = pxc.Vode(get_vode_energy(energy_type))
        self.vode_out = pxc.Vode(nudged_ce_energy(nudging))
        self.vode_out.h.frozen = True

        # Cached input embedding
        self.x_inj_cache = pxc.VodeParam()
        self.x_inj_cache.frozen = True

    def embed(self, x: jax.Array) -> jax.Array:
        x_inj = self.gn_in(self.input_conv(x))
        self.x_inj_cache.set(x_inj)
        return x_inj

    def __call__(self, y: jax.Array | None = None) -> jax.Array:
        x_inj = self.x_inj_cache.get()
        z = self.vode_z.get("h")
        self.vode_z(self.f(z, x_inj))
        z_out = self.pool(self.gn_out(z)).flatten()
        logits = self.linear(z_out)
        self.vode_out(logits)
        if y is not None:
            self.vode_out.set("h", y)
        return logits

    def forward_full(self, x: jax.Array, y: jax.Array | None = None) -> jax.Array:
        self.embed(x)
        return self(y)


# ======================================================================
# Multi-Vode DEQ  (2-node cycle: z1 ↔ z2)
# ======================================================================

class MultiVodeDEQPCModel(pxc.EnergyModule):
    """Two-node cyclic DEQ: z1* = f2(f1(z1*, x_inj), x_inj).

    Cycle:  z1 →[f1]→ z2 →[f2]→ z1
    Energy: E_diff(z2, f1(z1,x)) + E_diff(z1, f2(z2,x)) + ν·CE
    Readout from z1 → pool → linear → logits.
    """

    def __init__(
        self,
        n_channels: int = 48,
        n_classes: int = 10,
        nudging: float = 0.1,
        init_scale: float = 0.005,
        stop_grad_f: bool = False,
        energy_type: str = "se",
    ):
        super().__init__()
        self.n_classes = px.static(n_classes)
        self.n_channels = px.static(n_channels)
        self.nudging = px.static(nudging)
        self.init_scale = px.static(init_scale)
        self.stop_grad_f = px.static(stop_grad_f)
        self.energy_type = px.static(energy_type)

        # Encoder
        self.input_conv = pxnn.Conv2d(3, n_channels, 3, padding=(1, 1), use_bias=True)
        self.gn_in = GroupNorm(8, n_channels)

        # Two implicit blocks (same shape, different weights)
        self.f1 = ConvBlock(n_channels, init_scale=init_scale)
        self.f2 = ConvBlock(n_channels, init_scale=init_scale)

        # Readout (taps z1)
        self.gn_out = GroupNorm(8, n_channels)
        self.pool = pxnn.AvgPool2d(kernel_size=8, stride=8)
        self.linear = pxnn.Linear(n_channels * 4 * 4, n_classes)

        # Vodes — latent nodes use energy_type, output uses nudged CE
        vode_e = get_vode_energy(energy_type)
        self.vode_z1 = pxc.Vode(vode_e)
        self.vode_z2 = pxc.Vode(vode_e)
        self.vode_out = pxc.Vode(nudged_ce_energy(nudging))
        self.vode_out.h.frozen = True

        # Cached input embedding
        self.x_inj_cache = pxc.VodeParam()
        self.x_inj_cache.frozen = True

    def embed(self, x: jax.Array) -> jax.Array:
        x_inj = self.gn_in(self.input_conv(x))
        self.x_inj_cache.set(x_inj)
        return x_inj

    def __call__(self, y: jax.Array | None = None) -> jax.Array:
        x_inj = self.x_inj_cache.get()
        z1 = self.vode_z1.get("h")
        z2 = self.vode_z2.get("h")

        self.vode_z2(self.f1(z1, x_inj))   # u_z2 = f1(z1, x)
        self.vode_z1(self.f2(z2, x_inj))   # u_z1 = f2(z2, x)

        z_out = self.pool(self.gn_out(z1)).flatten()
        logits = self.linear(z_out)
        self.vode_out(logits)
        if y is not None:
            self.vode_out.set("h", y)
        return logits

    def forward_full(self, x: jax.Array, y: jax.Array | None = None) -> jax.Array:
        self.embed(x)
        return self(y)
