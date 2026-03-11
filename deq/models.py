"""DEQ-PC model definitions with a common latent interface.

All models implement:
    get_latent_vodes()  → list of latent Vode objects
    get_latents()       → batched latent tensor  (B, ...)
    set_latents(z)      → unpack and set on vodes
    latent_energy(z, x_inj, label)  → per-sample scalar energy
    free_energy(z, x_inj)           → per-sample scalar energy (no readout)

This allows a single, model-agnostic training loop.

Both ``energy_type`` options (``"se"`` / ``"log_cosh"``) are resolved
at JIT trace time via ``px.static``, so the compiled code contains
only the chosen branch.
"""

import jax
import jax.numpy as jnp

import pcx as px
import pcx.nn as pxnn
import pcx.predictive_coding as pxc

from .layers import GroupNorm, ResNetLayer, ConvBlock, FCBlock
from .energies import nudged_ce_energy, get_vode_energy, get_diff_energy


# ======================================================================
#  Shared helpers
# ======================================================================

def _readout_energy(z_out_flat, model, label):
    """CE readout energy, shared by all models."""
    logits = model.linear(z_out_flat)
    nudging = model.nudging.get()
    return (nudging * (-(label * jax.nn.log_softmax(logits)))).sum()


# ======================================================================
# Single-Vode DEQ
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

        # Encoder  (32×32 → 16×16)
        self.input_conv = pxnn.Conv2d(3, n_channels, 3, stride=2,
                                      padding=(1, 1), use_bias=True)
        self.gn_in = GroupNorm(8, n_channels)

        # Implicit block
        self.f = ResNetLayer(n_channels, n_inner, init_scale=init_scale)

        # Readout
        self.gn_out = GroupNorm(8, n_channels)
        self.pool = pxnn.AvgPool2d(kernel_size=4, stride=4)
        self.linear = pxnn.Linear(n_channels * 4 * 4, n_classes)

        # Vodes
        self.vode_z = pxc.Vode(get_vode_energy(energy_type))
        self.vode_out = pxc.Vode(nudged_ce_energy(nudging))
        self.vode_out.h.frozen = True

        self.x_inj_cache = pxc.VodeParam()
        self.x_inj_cache.frozen = True

    # ---- common interface ------------------------------------------------

    def get_latent_vodes(self):
        return [self.vode_z]

    def get_latents(self):
        return self.vode_z.h.get()

    def set_latents(self, z):
        self.vode_z.h.set(z)

    def free_energy(self, z, x_inj):
        """Per-sample internal consistency energy (no readout)."""
        u_z = self.f(z, x_inj)
        if self.stop_grad_f.get():
            u_z = jax.lax.stop_gradient(u_z)
        diff_energy_fn = get_diff_energy(self.energy_type.get())
        return diff_energy_fn(z - u_z)

    def latent_energy(self, z, x_inj, label):
        """Per-sample energy (called inside vmap(grad(...)))."""
        u_z = self.f(z, x_inj)
        if self.stop_grad_f.get():
            u_z = jax.lax.stop_gradient(u_z)

        diff_energy_fn = get_diff_energy(self.energy_type.get())
        e_z = diff_energy_fn(z - u_z)

        z_out = self.pool(self.gn_out(z)).flatten()
        e_out = _readout_energy(z_out, self, label)
        return e_z + e_out

    # ---- forward ---------------------------------------------------------

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

    def forward_full(self, x, y=None):
        self.embed(x)
        return self(y)


# ======================================================================
# Multi-Vode DEQ  (2-node cycle, simple encoder)
# ======================================================================

class MultiVodeDEQPCModel(pxc.EnergyModule):
    """Two-node cycle with a single-conv encoder (stride-2, 32→16)."""

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

        # Encoder  (32×32 → 16×16)
        self.input_conv = pxnn.Conv2d(3, n_channels, 3, stride=2,
                                      padding=(1, 1), use_bias=True)
        self.gn_in = GroupNorm(8, n_channels)

        # Implicit blocks
        self.f1 = ConvBlock(n_channels, init_scale=init_scale)
        self.f2 = ConvBlock(n_channels, init_scale=init_scale)

        # Readout (from z1)
        self.gn_out = GroupNorm(8, n_channels)
        self.pool = pxnn.AvgPool2d(kernel_size=4, stride=4)
        self.linear = pxnn.Linear(n_channels * 4 * 4, n_classes)

        # Vodes
        vode_e = get_vode_energy(energy_type)
        self.vode_z1 = pxc.Vode(vode_e)
        self.vode_z2 = pxc.Vode(vode_e)
        self.vode_out = pxc.Vode(nudged_ce_energy(nudging))
        self.vode_out.h.frozen = True

        self.x_inj_cache = pxc.VodeParam()
        self.x_inj_cache.frozen = True

    # ---- common interface ------------------------------------------------

    def get_latent_vodes(self):
        return [self.vode_z1, self.vode_z2]

    def get_latents(self):
        return jnp.concatenate(
            [self.vode_z1.h.get(), self.vode_z2.h.get()], axis=1
        )

    def set_latents(self, z):
        n_ch = self.n_channels.get()
        self.vode_z1.h.set(z[:, :n_ch])
        self.vode_z2.h.set(z[:, n_ch:])

    def free_energy(self, z, x_inj):
        """Per-sample internal consistency energy (no readout)."""
        n_ch = self.n_channels.get()
        z1, z2 = z[:n_ch], z[n_ch:]

        pred_z2 = self.f1(z1, x_inj)
        pred_z1 = self.f2(z2, x_inj)
        if self.stop_grad_f.get():
            pred_z2 = jax.lax.stop_gradient(pred_z2)
            pred_z1 = jax.lax.stop_gradient(pred_z1)

        diff_energy_fn = get_diff_energy(self.energy_type.get())
        return diff_energy_fn(z1 - pred_z1) + diff_energy_fn(z2 - pred_z2)

    def latent_energy(self, z, x_inj, label):
        n_ch = self.n_channels.get()
        z1, z2 = z[:n_ch], z[n_ch:]

        pred_z2 = self.f1(z1, x_inj)
        pred_z1 = self.f2(z2, x_inj)
        if self.stop_grad_f.get():
            pred_z2 = jax.lax.stop_gradient(pred_z2)
            pred_z1 = jax.lax.stop_gradient(pred_z1)

        diff_energy_fn = get_diff_energy(self.energy_type.get())
        e_z = diff_energy_fn(z1 - pred_z1) + diff_energy_fn(z2 - pred_z2)

        z_out = self.pool(self.gn_out(z1)).flatten()
        e_out = _readout_energy(z_out, self, label)
        return e_z + e_out

    # ---- forward ---------------------------------------------------------

    def embed(self, x):
        x_inj = self.gn_in(self.input_conv(x))
        self.x_inj_cache.set(x_inj)
        return x_inj

    def __call__(self, y=None):
        x_inj = self.x_inj_cache.get()
        z1 = self.vode_z1.get("h")
        z2 = self.vode_z2.get("h")

        self.vode_z2(self.f1(z1, x_inj))
        self.vode_z1(self.f2(z2, x_inj))

        z_out = self.pool(self.gn_out(z1)).flatten()
        logits = self.linear(z_out)
        self.vode_out(logits)
        if y is not None:
            self.vode_out.set("h", y)
        return logits

    def forward_full(self, x, y=None):
        self.embed(x)
        return self(y)


# ======================================================================
# Deep-encoder Multi-Vode DEQ  (2-node cycle, richer encoder)
# ======================================================================

class DeepDEQPCModel(pxc.EnergyModule):
    """Two-node cycle with a deeper encoder (3→64→128→n_ch, 32→16)."""

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

        # Encoder: 3→64 (32×32) → 128 (16×16) → n_channels (16×16)
        self.enc1 = pxnn.Conv2d(3, 64, 3, padding=(1, 1), use_bias=False)
        self.enc_norm1 = GroupNorm(8, 64)
        self.enc2 = pxnn.Conv2d(64, 128, 3, stride=2, padding=(1, 1),
                                use_bias=False)
        self.enc_norm2 = GroupNorm(8, 128)
        self.enc_proj = pxnn.Conv2d(128, n_channels, 1, use_bias=False)
        self.enc_proj_norm = GroupNorm(8, n_channels)

        # Implicit blocks
        self.f1 = ConvBlock(n_channels, init_scale=init_scale)
        self.f2 = ConvBlock(n_channels, init_scale=init_scale)

        # Readout (from z1)
        self.gn_out = GroupNorm(8, n_channels)
        self.pool = pxnn.AvgPool2d(kernel_size=4, stride=4)
        self.linear = pxnn.Linear(n_channels * 4 * 4, n_classes)

        # Vodes
        vode_e = get_vode_energy(energy_type)
        self.vode_z1 = pxc.Vode(vode_e)
        self.vode_z2 = pxc.Vode(vode_e)
        self.vode_out = pxc.Vode(nudged_ce_energy(nudging))
        self.vode_out.h.frozen = True

        self.x_inj_cache = pxc.VodeParam()
        self.x_inj_cache.frozen = True

    # ---- common interface ------------------------------------------------

    def get_latent_vodes(self):
        return [self.vode_z1, self.vode_z2]

    def get_latents(self):
        return jnp.concatenate(
            [self.vode_z1.h.get(), self.vode_z2.h.get()], axis=1
        )

    def set_latents(self, z):
        n_ch = self.n_channels.get()
        self.vode_z1.h.set(z[:, :n_ch])
        self.vode_z2.h.set(z[:, n_ch:])

    def free_energy(self, z, x_inj):
        """Per-sample internal consistency energy (no readout)."""
        n_ch = self.n_channels.get()
        z1, z2 = z[:n_ch], z[n_ch:]

        pred_z2 = self.f1(z1, x_inj)
        pred_z1 = self.f2(z2, x_inj)
        if self.stop_grad_f.get():
            pred_z2 = jax.lax.stop_gradient(pred_z2)
            pred_z1 = jax.lax.stop_gradient(pred_z1)

        diff_energy_fn = get_diff_energy(self.energy_type.get())
        return diff_energy_fn(z1 - pred_z1) + diff_energy_fn(z2 - pred_z2)

    def latent_energy(self, z, x_inj, label):
        n_ch = self.n_channels.get()
        z1, z2 = z[:n_ch], z[n_ch:]

        pred_z2 = self.f1(z1, x_inj)
        pred_z1 = self.f2(z2, x_inj)
        if self.stop_grad_f.get():
            pred_z2 = jax.lax.stop_gradient(pred_z2)
            pred_z1 = jax.lax.stop_gradient(pred_z1)

        diff_energy_fn = get_diff_energy(self.energy_type.get())
        e_z = diff_energy_fn(z1 - pred_z1) + diff_energy_fn(z2 - pred_z2)

        z_out = self.pool(self.gn_out(z1)).flatten()
        e_out = _readout_energy(z_out, self, label)
        return e_z + e_out

    # ---- forward ---------------------------------------------------------

    def embed(self, x):
        h = jax.nn.relu(self.enc_norm1(self.enc1(x)))
        h = jax.nn.relu(self.enc_norm2(self.enc2(h)))
        x_inj = self.enc_proj_norm(self.enc_proj(h))
        self.x_inj_cache.set(x_inj)
        return x_inj

    def __call__(self, y=None):
        x_inj = self.x_inj_cache.get()
        z1 = self.vode_z1.get("h")
        z2 = self.vode_z2.get("h")

        self.vode_z2(self.f1(z1, x_inj))
        self.vode_z1(self.f2(z2, x_inj))

        z_out = self.pool(self.gn_out(z1)).flatten()
        logits = self.linear(z_out)
        self.vode_out(logits)
        if y is not None:
            self.vode_out.set("h", y)
        return logits

    def forward_full(self, x, y=None):
        self.embed(x)
        return self(y)



# ======================================================================
# FC Multi-Vode DEQ
# ======================================================================

class FCMultiVodeDEQPCModel(pxc.EnergyModule):
    """Two-node FC implicit model for flat-input datasets.

    Architecture
    ------------
    Encoder:   Linear(input_dim → hidden_dim) + LayerNorm
    Implicit:  FCBlock f1 (z1 → pred_z2), FCBlock f2 (z2 → pred_z1)
    Readout:   LayerNorm + Linear(hidden_dim → n_classes)

    Implements the same common interface as the CNN multi-vode models:
        get_latent_vodes, get_latents, set_latents,
        latent_energy, free_energy, embed, forward_full.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        input_dim: int = 784,
        n_classes: int = 10,
        nudging: float = 0.1,
        init_scale: float = 0.005,
        stop_grad_f: bool = False,
        energy_type: str = "se",
    ):
        super().__init__()
        self.n_classes = px.static(n_classes)
        self.hidden_dim = px.static(hidden_dim)
        self.nudging = px.static(nudging)
        self.init_scale = px.static(init_scale)
        self.stop_grad_f = px.static(stop_grad_f)
        self.energy_type = px.static(energy_type)

        # Encoder: flatten → hidden_dim
        self.input_linear = pxnn.Linear(input_dim, hidden_dim, bias=True)
        self.ln_in = pxnn.LayerNorm((hidden_dim,))

        # Two implicit FC blocks (the fixed-point cycle)
        self.f1 = FCBlock(hidden_dim, init_scale=init_scale)
        self.f2 = FCBlock(hidden_dim, init_scale=init_scale)

        # Readout head (from z1)
        self.ln_out = pxnn.LayerNorm((hidden_dim,))
        self.linear = pxnn.Linear(hidden_dim, n_classes)

        # Predictive coding nodes
        vode_e = get_vode_energy(energy_type)
        self.vode_z1 = pxc.Vode(vode_e)
        self.vode_z2 = pxc.Vode(vode_e)
        self.vode_out = pxc.Vode(nudged_ce_energy(nudging))
        self.vode_out.h.frozen = True          # output target is clamped

        self.x_inj_cache = pxc.VodeParam()
        self.x_inj_cache.frozen = True         # embedding is not updated by inference

    # ---- common interface ------------------------------------------------

    def get_latent_vodes(self):
        return [self.vode_z1, self.vode_z2]

    def get_latents(self):
        """Return concatenated latents: (B, 2 * hidden_dim)."""
        return jnp.concatenate(
            [self.vode_z1.h.get(), self.vode_z2.h.get()], axis=1
        )

    def set_latents(self, z):
        """Unpack concatenated latents back into the two vodes."""
        dim = self.hidden_dim.get()
        self.vode_z1.h.set(z[:, :dim])
        self.vode_z2.h.set(z[:, dim:])

    def free_energy(self, z, x_inj):
        """Per-sample internal consistency energy (no readout).

        Called inside vmap(grad(...)), so z has shape (2*hidden_dim,)
        and x_inj has shape (hidden_dim,).
        """
        dim = self.hidden_dim.get()
        z1, z2 = z[:dim], z[dim:]

        pred_z2 = self.f1(z1, x_inj)
        pred_z1 = self.f2(z2, x_inj)
        if self.stop_grad_f.get():
            pred_z2 = jax.lax.stop_gradient(pred_z2)
            pred_z1 = jax.lax.stop_gradient(pred_z1)

        diff_energy_fn = get_diff_energy(self.energy_type.get())
        return diff_energy_fn(z1 - pred_z1) + diff_energy_fn(z2 - pred_z2)

    def latent_energy(self, z, x_inj, label):
        """Per-sample full energy = internal consistency + CE readout."""
        dim = self.hidden_dim.get()
        z1, z2 = z[:dim], z[dim:]

        pred_z2 = self.f1(z1, x_inj)
        pred_z1 = self.f2(z2, x_inj)
        if self.stop_grad_f.get():
            pred_z2 = jax.lax.stop_gradient(pred_z2)
            pred_z1 = jax.lax.stop_gradient(pred_z1)

        diff_energy_fn = get_diff_energy(self.energy_type.get())
        e_z = diff_energy_fn(z1 - pred_z1) + diff_energy_fn(z2 - pred_z2)

        z_out = self.ln_out(z1)
        e_out = _readout_energy(z_out, self, label)
        return e_z + e_out

    # ---- forward ---------------------------------------------------------

    def embed(self, x: jax.Array) -> jax.Array:
        """Encode one sample: (1, 28, 28) → (hidden_dim,)."""
        x_inj = self.ln_in(self.input_linear(x.flatten()))
        self.x_inj_cache.set(x_inj)
        return x_inj

    def __call__(self, y=None):
        x_inj = self.x_inj_cache.get()
        z1 = self.vode_z1.get("h")
        z2 = self.vode_z2.get("h")

        # Cycle: f1 predicts z2, f2 predicts z1
        self.vode_z2(self.f1(z1, x_inj))
        self.vode_z1(self.f2(z2, x_inj))

        # Readout from z1
        z_out = self.ln_out(z1)
        logits = self.linear(z_out)
        self.vode_out(logits)
        if y is not None:
            self.vode_out.set("h", y)
        return logits

    def forward_full(self, x, y=None):
        self.embed(x)
        return self(y)