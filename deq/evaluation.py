"""High-level evaluation utilities, model-agnostic."""

import time
from typing import Callable

import jax
import numpy as np


def evaluate_accuracy(
    dl,
    T_steps: int,
    *,
    model,
    optim_h,
    eval_fn: Callable,
    lr_decay: float = 1.0,
    max_samples: int | None = None,
) -> float:
    """Run *eval_fn* over a dataloader and return mean accuracy."""
    accs = []
    seen = 0
    n_classes = int(model.n_classes.get())
    for x, y in dl:
        y_oh = jax.nn.one_hot(y.numpy(), n_classes)
        accs.append(eval_fn(T_steps, x.numpy(), y_oh, lr_decay,
                            model=model, optim_h=optim_h))
        seen += x.shape[0]
        if max_samples is not None and seen >= max_samples:
            break
    return float(np.mean(accs))


def evaluate_epoch(
    test_dl,
    T_steps: int,
    *,
    model,
    optim_h,
    eval_fn: Callable,
    lr_decay: float = 1.0,
):
    """Evaluate on test split and return (test_acc, wall_time)."""
    t0 = time.perf_counter()

    test_acc = evaluate_accuracy(
        test_dl, T_steps, model=model, optim_h=optim_h,
        eval_fn=eval_fn, lr_decay=lr_decay,
    )
    return test_acc, time.perf_counter() - t0