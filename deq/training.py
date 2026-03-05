"""Inference loops and per-batch train / eval functions for DEQ-PC models.

The energy used inside each inference loop is selected via
``model.energy_type.get()`` — a ``px.static`` value resolved at JIT
trace time, so the compiled code contains **only** the chosen branch.

Public API
----------
Single-vode  : train_on_batch,       eval_on_batch
Multi-vode   : train_on_batch_multi, eval_on_batch_multi
"""

import jax
import jax.numpy as jnp
import optax

import pcx.functional as pxf
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

from .models import DEQPCModel, MultiVodeDEQPCModel
from .energies import get_diff_energy

# Masks shared by both model families
_V = pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0))
_w_mask = pxu.M(pxnn.LayerParam).to([False, True])


# ======================================================================
#  Single-vode helpers
# ======================================================================

@pxf.vmap(_V, in_axes=(0, 0), out_axes=0)
def _init_train(x, y, *, model: DEQPCModel):
    x_inj = model.embed(x)
    model.vode_z.h.set(jnp.zeros_like(x_inj))
    return model(y)


@pxf.vmap(_V, in_axes=(0,), out_axes=0)
def _init_eval(x, *, model: DEQPCModel):
    x_inj = model.embed(x)
    model.vode_z.h.set(jnp.zeros_like(x_inj))
    model.vode_out.h.set(jnp.zeros(model.n_classes.get()))
    return model()


@pxf.vmap(_V, in_axes=(0,), out_axes=None, axis_name="batch")
def _energy_weight(x, *, model: DEQPCModel):
    model.forward_full(x)
    return jax.lax.psum(model.energy(), "batch")


@pxf.vmap(_V, in_axes=(), out_axes=0)
def _predict_cached(*, model: DEQPCModel):
    return model()


def _run_inference_loop(T, z_h, x_inj, label_h, h_opt, model):
    """Minimise E(z) for *T* steps.  Energy type is read from model (static)."""
    h_opt_state = h_opt.init(z_h)
    nudging = model.nudging.get()
    stop = model.stop_grad_f.get()
    diff_energy_fn = get_diff_energy(model.energy_type.get())

    #@jax.checkpoint
    def per_sample_energy(z, x_i, lab):
        u_z = model.f(z, x_i)
        if stop:
            u_z = jax.lax.stop_gradient(u_z)
        e_z = diff_energy_fn(z - u_z)
        z_out = model.pool(model.gn_out(z)).flatten()
        logits = model.linear(z_out)
        e_out = (nudging * (-(lab * jax.nn.log_softmax(logits)))).sum()
        return e_z + e_out

    grad_fn = jax.vmap(jax.grad(per_sample_energy))

    def body(_, carry):
        z_h, opt_state = carry
        g = grad_fn(z_h, x_inj, label_h)
        updates, opt_state = h_opt.update(g, opt_state, z_h)
        z_h = optax.apply_updates(z_h, updates)
        return z_h, opt_state

    z_h, _ = jax.lax.fori_loop(0, T, body, (z_h, h_opt_state))
    return z_h


@pxf.jit(static_argnums=0)
def train_on_batch(T, x, y, *, model: DEQPCModel, optim_w, optim_h):
    model.train()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x_inj = jax.vmap(lambda xi: model.gn_in(model.input_conv(xi)))(x)
        model.x_inj_cache.set(x_inj)
        model.vode_z.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(y)

    h_opt = optim_h.optax_opt_fn()
    z_h = _run_inference_loop(
        T, model.vode_z.h.get(), model.x_inj_cache.get(),
        model.vode_out.h.get(), h_opt, model,
    )
    model.vode_z.h.set(z_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, grads = pxf.value_and_grad(_w_mask, has_aux=False)(_energy_weight)(
            x, model=model,
        )
    optim_w.step(model, grads["model"], scale_by=1.0 / x.shape[0])


@pxf.jit(static_argnums=0)
def eval_on_batch(T, x, y_oh, *, model: DEQPCModel, optim_h):
    model.eval()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x_inj = jax.vmap(lambda xi: model.gn_in(model.input_conv(xi)))(x)
        model.x_inj_cache.set(x_inj)
        model.vode_z.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(jnp.zeros_like(y_oh))

    h_opt = optim_h.optax_opt_fn()
    z_h = _run_inference_loop(
        T, model.vode_z.h.get(), model.x_inj_cache.get(),
        jnp.zeros_like(y_oh), h_opt, model,
    )
    model.vode_z.h.set(z_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        y_pred = _predict_cached(model=model).argmax(axis=-1)
    return (y_pred == y_oh.argmax(axis=-1)).mean()


# ======================================================================
#  Multi-vode helpers  (2-node cycle)
# ======================================================================

@pxf.vmap(_V, in_axes=(0, 0), out_axes=0)
def _init_train_multi(x, y, *, model: MultiVodeDEQPCModel):
    x_inj = model.embed(x)
    model.vode_z1.h.set(jnp.zeros_like(x_inj))
    model.vode_z2.h.set(jnp.zeros_like(x_inj))
    return model(y)


@pxf.vmap(_V, in_axes=(0,), out_axes=0)
def _init_eval_multi(x, *, model: MultiVodeDEQPCModel):
    x_inj = model.embed(x)
    model.vode_z1.h.set(jnp.zeros_like(x_inj))
    model.vode_z2.h.set(jnp.zeros_like(x_inj))
    model.vode_out.h.set(jnp.zeros(model.n_classes.get()))
    return model()


@pxf.vmap(_V, in_axes=(0,), out_axes=None, axis_name="batch")
def _energy_weight_multi(x, *, model: MultiVodeDEQPCModel):
    model.forward_full(x)
    return jax.lax.psum(model.energy(), "batch")


@pxf.vmap(_V, in_axes=(), out_axes=0)
def _predict_cached_multi(*, model: MultiVodeDEQPCModel):
    return model()


def _run_inference_loop_multi(T, z1_h, z2_h, x_inj, label_h, h_opt, model):
    """Joint minimisation of E(z1, z2) for *T* steps.

    z1 and z2 are concatenated along the channel axis so that a single
    optax state covers both.  Energy type is read from model (static).
    """
    n_ch = z1_h.shape[1]
    z_h = jnp.concatenate([z1_h, z2_h], axis=1)          # (B, 2C, H, W)
    h_opt_state = h_opt.init(z_h)

    nudging = model.nudging.get()
    stop = model.stop_grad_f.get()
    diff_energy_fn = get_diff_energy(model.energy_type.get())

    #@jax.checkpoint
    def per_sample_energy(z, x_i, lab):
        z1 = z[:n_ch]
        z2 = z[n_ch:]

        pred_z2 = model.f1(z1, x_i)
        pred_z1 = model.f2(z2, x_i)

        if stop:
            pred_z2 = jax.lax.stop_gradient(pred_z2)
            pred_z1 = jax.lax.stop_gradient(pred_z1)

        e_z = diff_energy_fn(z1 - pred_z1) + diff_energy_fn(z2 - pred_z2)

        z_out = model.pool(model.gn_out(z1)).flatten()
        logits = model.linear(z_out)
        e_out = (nudging * (-(lab * jax.nn.log_softmax(logits)))).sum()
        return e_z + e_out

    grad_fn = jax.vmap(jax.grad(per_sample_energy))

    def body(_, carry):
        z_h, opt_state = carry
        g = grad_fn(z_h, x_inj, label_h)
        updates, opt_state = h_opt.update(g, opt_state, z_h)
        z_h = optax.apply_updates(z_h, updates)
        return z_h, opt_state

    z_h, _ = jax.lax.fori_loop(0, T, body, (z_h, h_opt_state))
    return z_h[:, :n_ch], z_h[:, n_ch:]


@pxf.jit(static_argnums=0)
def train_on_batch_multi(T, x, y, *, model: MultiVodeDEQPCModel, optim_w, optim_h):
    model.train()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x_inj = jax.vmap(lambda xi: model.gn_in(model.input_conv(xi)))(x)
        model.x_inj_cache.set(x_inj)
        model.vode_z1.h.set(jnp.zeros_like(x_inj))
        model.vode_z2.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(y)

    h_opt = optim_h.optax_opt_fn()
    z1_h, z2_h = _run_inference_loop_multi(
        T,
        model.vode_z1.h.get(), model.vode_z2.h.get(),
        model.x_inj_cache.get(), model.vode_out.h.get(),
        h_opt, model,
    )
    model.vode_z1.h.set(z1_h)
    model.vode_z2.h.set(z2_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, grads = pxf.value_and_grad(_w_mask, has_aux=False)(_energy_weight_multi)(
            x, model=model,
        )
    optim_w.step(model, grads["model"], scale_by=1.0 / x.shape[0])


@pxf.jit(static_argnums=0)
def eval_on_batch_multi(T, x, y_oh, *, model: MultiVodeDEQPCModel, optim_h):
    model.eval()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x_inj = jax.vmap(lambda xi: model.gn_in(model.input_conv(xi)))(x)
        model.x_inj_cache.set(x_inj)
        model.vode_z1.h.set(jnp.zeros_like(x_inj))
        model.vode_z2.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(jnp.zeros_like(y_oh))

    h_opt = optim_h.optax_opt_fn()
    
    z1_h, z2_h = _run_inference_loop_multi(
        T,
        model.vode_z1.h.get(), model.vode_z2.h.get(),
        model.x_inj_cache.get(), jnp.zeros_like(y_oh),
        h_opt, model,
    )
    model.vode_z1.h.set(z1_h)
    model.vode_z2.h.set(z2_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        y_pred = _predict_cached_multi(model=model).argmax(axis=-1)
    return (y_pred == y_oh.argmax(axis=-1)).mean()
