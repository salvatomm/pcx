"""Spectral normalization for DEQ-PC implicit block convolutions.

Warm-starts the singular vector across calls (like PyTorch's spectral_norm),
so a single power iteration per step suffices. JIT-compiled core.

Usage:
    from deq.spectral_norm import SpectralProjector

    projector = SpectralProjector(model, target=0.95)

    for epoch in range(N_EPOCHS):
        for x, y in train_dl:
            train_on_batch(...)
            projector.project()           # 1 power-iter step, near-zero cost
"""

import jax
import jax.numpy as jnp

import pcx.nn as pxnn


@jax.jit
def _project_weight(w, v, target):
    """Single warm-started power iteration + conditional projection.

    Args:
        w: conv weight (out, in, kH, kW)
        v: right singular vector from previous call (in*kH*kW,)
        target: spectral norm upper bound

    Returns:
        (w_new, v_new, sigma)
    """
    h = w.reshape(w.shape[0], -1)          # (out, fan_in)

    # One power iteration (warm-started, so one step tracks well)
    u = h @ v
    u = u / (jnp.linalg.norm(u) + 1e-12)
    v_new = h.T @ u
    v_new = v_new / (jnp.linalg.norm(v_new) + 1e-12)

    sigma = u @ h @ v_new

    w_new = jnp.where(sigma > target, w * (target / sigma), w)
    return w_new, v_new, sigma


class SpectralProjector:
    """Maintains per-conv singular vectors for warm-started projection."""

    def __init__(self, model, target: float = 0.95):
        self.target = target
        self._entries = []  # list of [LayerParam, v_vector]

        # Collect implicit-block convs
        convs = []
        if hasattr(model, 'f') and not hasattr(model, 'f1'):
            convs.extend([model.f.conv1, model.f.conv2])
        if hasattr(model, 'f1'):
            convs.append(model.f1.conv)
        if hasattr(model, 'f2'):
            convs.append(model.f2.conv)

        for conv in convs:
            wp = conv.nn.weight
            w = wp.get()
            fan_in = w.reshape(w.shape[0], -1).shape[1]
            v = jnp.ones((fan_in,)) / jnp.sqrt(fan_in)
            self._entries.append([wp, v])

    def project(self):
        """Run one power-iteration step and project if needed."""
        for entry in self._entries:
            wp, v = entry
            w_new, v_new, _ = _project_weight(wp.get(), v, self.target)
            wp.set(w_new)
            entry[1] = v_new