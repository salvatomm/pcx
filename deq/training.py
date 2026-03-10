"""Model-agnostic inference loops and per-batch train / eval for DEQ-PC.

Works with any model that implements the common interface:
    get_latent_vodes(), get_latents(), set_latents(z),
    latent_energy(z, x, y), free_energy(z, x)

Public API
----------
    train_on_batch   (T, x, y, lr_decay, *, model, optim_w, optim_h)
    eval_on_batch    (T, x, y_oh, lr_decay, *, model, optim_h)
    train_on_batch_ep(T_free, T_nudg, x, y, lr_decay, *, model, optim_w, optim_h)
    eval_on_batch_ep (T_free, T_nudg, x, y_oh, lr_decay, *, model, optim_h)
"""

import jax
import jax.numpy as jnp
import optax

import pcx.functional as pxf
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

# ── Masks (shared across all models) ──────────────────────────────────

_V = pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0))
_w_mask = pxu.M(pxnn.LayerParam).to([False, True])


# ── Standard inference loop ───────────────────────────────────────────

def _run_inference_loop(T, z_h, x_inj, label_h, h_opt, model, lr_decay):
    """Minimise E(z) for *T* steps via the model's ``latent_energy``.

    At iteration i the effective learning rate is lr * lr_decay**i.
    Since we cannot mutate the optax optimizer's lr inside fori_loop,
    we scale the *updates* instead — mathematically identical.
    """
    h_opt_state = h_opt.init(z_h)

    energy_fn = lambda z, xi, lab: model.latent_energy(z, xi, lab)
    grad_fn = jax.vmap(jax.grad(energy_fn))

    def body(i, carry):
        z_h, opt_state = carry
        g = grad_fn(z_h, x_inj, label_h)
        updates, opt_state = h_opt.update(g, opt_state, z_h)
        updates = updates * (lr_decay ** i)
        z_h = optax.apply_updates(z_h, updates)
        return z_h, opt_state

    z_h, _ = jax.lax.fori_loop(0, T, body, (z_h, h_opt_state))
    return z_h


# ── Equilibrium Propagation inference loop ────────────────────────────

def _run_ep_inference(T_free, T_nudg, z_h, x_inj, label_h, h_opt, model, lr_decay):
    """Two-phase equilibrium propagation inference.

    Phase 1 — Free (T_free steps):
        Minimize only internal consistency energy (``model.free_energy``).
        No readout, no label, no linear layer in the gradient graph.

    Phase 2 — Nudged (T_nudg steps):
        Starting from the free-phase equilibrium, minimize the full energy
        (``model.latent_energy`` = internal + nudged CE readout).

    Optimizer state (momentum buffers) threads across both phases.
    LR decay is continuous: nudged phase starts at ``lr_decay ** T_free``.
    """
    h_opt_state = h_opt.init(z_h)

    # --- Free phase: only internal consistency, no readout ---
    free_grad_fn = jax.vmap(jax.grad(
        lambda z, xi: model.free_energy(z, xi)
    ))

    def free_body(i, carry):
        z_h, opt_state = carry
        g = free_grad_fn(z_h, x_inj)
        updates, opt_state = h_opt.update(g, opt_state, z_h)
        updates = updates * (lr_decay ** i)
        z_h = optax.apply_updates(z_h, updates)
        return z_h, opt_state

    z_h, h_opt_state = jax.lax.fori_loop(
        0, T_free, free_body, (z_h, h_opt_state)
    )

    # --- Nudged phase: full energy with readout ---
    nudged_grad_fn = jax.vmap(jax.grad(
        lambda z, xi, lab: model.latent_energy(z, xi, lab)
    ))

    def nudged_body(i, carry):
        z_h, opt_state = carry
        g = nudged_grad_fn(z_h, x_inj, label_h)
        updates, opt_state = h_opt.update(g, opt_state, z_h)
        # LR decay continues from T_free
        updates = updates * (lr_decay ** (T_free + i))
        z_h = optax.apply_updates(z_h, updates)
        return z_h, opt_state

    z_h, _ = jax.lax.fori_loop(
        0, T_nudg, nudged_body, (z_h, h_opt_state)
    )
    return z_h


def _run_free_inference(T, z_h, x_inj, h_opt, model, lr_decay):
    """Pure free-phase inference (for eval). No label, no readout in the loop."""
    h_opt_state = h_opt.init(z_h)

    free_grad_fn = jax.vmap(jax.grad(
        lambda z, xi: model.free_energy(z, xi)
    ))

    def body(i, carry):
        z_h, opt_state = carry
        g = free_grad_fn(z_h, x_inj)
        updates, opt_state = h_opt.update(g, opt_state, z_h)
        updates = updates * (lr_decay ** i)
        z_h = optax.apply_updates(z_h, updates)
        return z_h, opt_state

    z_h, _ = jax.lax.fori_loop(0, T, body, (z_h, h_opt_state))
    return z_h


# ── Weight-gradient & prediction helpers ──────────────────────────────

@pxf.vmap(_V, in_axes=(0,), out_axes=None, axis_name="batch")
def _energy_weight(x, *, model):
    model.forward_full(x)
    return jax.lax.psum(model.energy(), "batch")


@pxf.vmap(_V, in_axes=(), out_axes=0)
def _predict_cached(*, model):
    return model()


# ── Shared init helper ────────────────────────────────────────────────

def _init_batch(x, y, model):
    """Embed input and zero-init all latent vodes + output vode."""
    x_inj = jax.vmap(lambda xi: model.embed(xi))(x)
    model.x_inj_cache.set(x_inj)
    for vode in model.get_latent_vodes():
        vode.h.set(jnp.zeros_like(x_inj))
    model.vode_out.h.set(y)


def _weight_step(x, model, optim_w):
    """Compute weight gradients and apply optimizer step."""
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, grads = pxf.value_and_grad(
            _w_mask, has_aux=False
        )(_energy_weight)(x, model=model)
    optim_w.step(model, grads["model"], scale_by=1.0 / x.shape[0])


# ── Standard Public API ───────────────────────────────────────────────

@pxf.jit(static_argnums=0)
def train_on_batch(T, x, y, lr_decay, *, model, optim_w, optim_h):
    model.train()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _init_batch(x, y, model)

    h_opt = optim_h.optax_opt_fn()
    z_h = _run_inference_loop(
        T, model.get_latents(), model.x_inj_cache.get(),
        model.vode_out.h.get(), h_opt, model, lr_decay,
    )
    model.set_latents(z_h)

    _weight_step(x, model, optim_w)


@pxf.jit(static_argnums=0)
def eval_on_batch(T, x, y_oh, lr_decay, *, model, optim_h):
    model.eval()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x_inj = jax.vmap(lambda xi: model.embed(xi))(x)
        model.x_inj_cache.set(x_inj)
        for vode in model.get_latent_vodes():
            vode.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(jnp.zeros_like(y_oh))

    h_opt = optim_h.optax_opt_fn()
    z_h = _run_inference_loop(
        T, model.get_latents(), model.x_inj_cache.get(),
        jnp.zeros_like(y_oh), h_opt, model, lr_decay,
    )
    model.set_latents(z_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        y_pred = _predict_cached(model=model).argmax(axis=-1)
    return (y_pred == y_oh.argmax(axis=-1)).mean()


# ── Equilibrium Propagation Public API ────────────────────────────────

@pxf.jit(static_argnums=(0, 1))
def train_on_batch_ep(T_free, T_nudg, x, y, lr_decay, *, model, optim_w, optim_h):
    """EP training: free phase finds the fixed point, nudged phase injects label."""
    model.train()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _init_batch(x, y, model)

    h_opt = optim_h.optax_opt_fn()
    z_h = _run_ep_inference(
        T_free, T_nudg,
        model.get_latents(), model.x_inj_cache.get(),
        model.vode_out.h.get(), h_opt, model, lr_decay,
    )
    model.set_latents(z_h)

    _weight_step(x, model, optim_w)


@pxf.jit(static_argnums=(0, 1))
def eval_on_batch_ep(T_free, T_nudg, x, y_oh, lr_decay, *, model, optim_h):
    """EP eval: pure free-phase for T_free + T_nudg steps, then readout."""
    model.eval()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x_inj = jax.vmap(lambda xi: model.embed(xi))(x)
        model.x_inj_cache.set(x_inj)
        for vode in model.get_latent_vodes():
            vode.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(jnp.zeros_like(y_oh))

    # At eval, run pure free-phase for the full budget (no label signal)
    T_total = T_free + T_nudg
    h_opt = optim_h.optax_opt_fn()
    z_h = _run_free_inference(
        T_total,
        model.get_latents(), model.x_inj_cache.get(),
        h_opt, model, lr_decay,
    )
    model.set_latents(z_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        y_pred = _predict_cached(model=model).argmax(axis=-1)
    return (y_pred == y_oh.argmax(axis=-1)).mean()