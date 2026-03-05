#!/usr/bin/env python3
"""Train a 2-vode (multi-node cycle) DEQ-PC model on CIFAR-10.

Usage
-----
    python -m deq.run_multi_vode                  # default: SE energy
    python -m deq.run_multi_vode --energy log_cosh # log-cosh energy
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import optax

import pcx as px
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

from deq.models import MultiVodeDEQPCModel
from deq.training import train_on_batch_multi, eval_on_batch_multi
from deq.evaluation import evaluate_epoch
from deq.data import get_dataloaders


# ── Hyperparameters ───────────────────────────────────────────────────
SEED = 0
N_EPOCHS = 30
T_STEPS = 100
N_CLASSES = 10
N_CHANNELS = 48

NUDGING = 0.1
INIT_SCALE = 0.005
STOP_GRAD_F = False

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 1000
TRAIN_EVAL_SAMPLES = 10_000
DATA_ROOT = "~/tmp/cifar10/"

LR_W = 5e-4
WEIGHT_DECAY_W = 5e-3
LR_H = 0.2
MOMENTUM_H = 0.5
ENERGY_FN = "log_cosh"  # "se" or "log_cosh"


# ── Utilities ─────────────────────────────────────────────────────────

def assert_gpu_backend():
    print(f"Python: {sys.executable}")
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend: {backend} | devices: {devices}")
    if backend != "gpu":
        raise RuntimeError("GPU backend required.")


def train_epoch(train_dl, *, model, optim_w, optim_h):
    for x, y in train_dl:
        train_on_batch_multi(
            T_STEPS,
            x.numpy(),
            jax.nn.one_hot(y.numpy(), N_CLASSES),
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
        )


# ── Main ──────────────────────────────────────────────────────────────

def main():

    energy_type = ENERGY_FN
    print(f"Energy type: {energy_type}")

    px.RKG.seed(SEED)
    assert_gpu_backend()

    train_dl, test_dl = get_dataloaders(
        TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, root=DATA_ROOT,
    )

    model = MultiVodeDEQPCModel(
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        nudging=NUDGING,
        init_scale=INIT_SCALE,
        stop_grad_f=STOP_GRAD_F,
        energy_type=energy_type,
    )

    optim_h = pxu.Optim(
        lambda: optax.sgd(LR_H, momentum=MOMENTUM_H, nesterov=True),
    )

    steps_per_epoch = len(train_dl)
    schedule_w = optax.piecewise_constant_schedule(
        init_value=LR_W,
        boundaries_and_scales={
            20 * steps_per_epoch: 0.2,
            40 * steps_per_epoch: 0.2,
        },
    )
    optim_w = pxu.Optim(
        lambda: optax.adamw(schedule_w, weight_decay=WEIGHT_DECAY_W),
        pxu.M(pxnn.LayerParam)(model),
    )

    # Warmup (compile / shape-init)
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x0 = jnp.zeros((TRAIN_BATCH_SIZE, 3, 32, 32))
        y0 = jnp.zeros((TRAIN_BATCH_SIZE, N_CLASSES))
        x_inj = jax.vmap(lambda xi: model.gn_in(model.input_conv(xi)))(x0)
        model.x_inj_cache.set(x_inj)
        model.vode_z1.h.set(jnp.zeros_like(x_inj))
        model.vode_z2.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(y0)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        train_epoch(train_dl, model=model, optim_w=optim_w, optim_h=optim_h)
        train_time = time.perf_counter() - t0

        test_acc, eval_time = evaluate_epoch(
            test_dl, T_STEPS,
            model=model, optim_h=optim_h,
            eval_fn=eval_on_batch_multi,
        )

        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | "
            f"Test {test_acc*100:5.2f}% | "
            f"Train {train_time:.1f}s | "
            f"Eval {eval_time:.1f}s"
        )


if __name__ == "__main__":
    main()
