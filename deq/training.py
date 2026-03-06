"""Model-agnostic inference loops and per-batch train / eval for DEQ-PC.

Works with any model that implements the common interface:
    get_latent_vodes(), get_latents(), set_latents(z), latent_energy(z, x, y)

Public API
----------
    train_on_batch(T, x, y, *, model, optim_w, optim_h)
    eval_on_batch (T, x, y_oh, *, model, optim_h)
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


# ── Inference loop ────────────────────────────────────────────────────

def _run_inference_loop(T, z_h, x_inj, label_h, h_opt, model):
    """Minimise E(z) for *T* steps via the model's ``latent_energy``."""
    h_opt_state = h_opt.init(z_h)

    energy_fn = lambda z, xi, lab: model.latent_energy(z, xi, lab)
    grad_fn = jax.vmap(jax.grad(energy_fn))

    def body(_, carry):
        z_h, opt_state = carry
        g = grad_fn(z_h, x_inj, label_h)
        updates, opt_state = h_opt.update(g, opt_state, z_h)
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


# ── Public API ────────────────────────────────────────────────────────

@pxf.jit(static_argnums=0)
def train_on_batch(T, x, y, *, model, optim_w, optim_h):
    model.train()

    # 1. Embed & initialise latents
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x_inj = jax.vmap(lambda xi: model.embed(xi))(x)
        model.x_inj_cache.set(x_inj)
        for vode in model.get_latent_vodes():
            vode.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(y)

    # 2. Inference: minimise latent energy
    h_opt = optim_h.optax_opt_fn()
    z_h = _run_inference_loop(
        T, model.get_latents(), model.x_inj_cache.get(),
        model.vode_out.h.get(), h_opt, model,
    )
    model.set_latents(z_h)

    # 3. Weight update
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, grads = pxf.value_and_grad(
            _w_mask, has_aux=False
        )(_energy_weight)(x, model=model)
    optim_w.step(model, grads["model"], scale_by=1.0 / x.shape[0])


@pxf.jit(static_argnums=0)
def eval_on_batch(T, x, y_oh, *, model, optim_h):
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
        jnp.zeros_like(y_oh), h_opt, model,
    )
    model.set_latents(z_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        y_pred = _predict_cached(model=model).argmax(axis=-1)
    return (y_pred == y_oh.argmax(axis=-1)).mean()