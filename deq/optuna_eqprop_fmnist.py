"""Optuna hyperparameter search for EP DEQ-PC on FashionMNIST.

This mirrors ``optuna_eqprop.py`` but switches to the FC two-node model
(``FCMultiVodeDEQPCModel``) and FashionMNIST data.

Compared to the CIFAR script, this search space is broader and includes
additional notebook-inspired knobs (hidden_dim, stop_grad_f, energy_type,
per-step latent LR decay, spectral projection toggle/target, schedule scales).
"""

import io
import os
import sys
import subprocess
import multiprocessing as mp
from contextlib import redirect_stderr, redirect_stdout

import optuna

SEED = 0
N_CLASSES = 10

BATCH_SIZE_CHOICES = [128, 256, 512]
HIDDEN_DIM_CHOICES = [64]
T_FREE_CHOICES = [0, 5, 10, 20, 30]
T_NUDG_CHOICES = [30, 40, 50, 60]

TEST_BATCH_SIZE = 1000
DATA_ROOT = "~/tmp/fmnist/"

N_EPOCHS_SEARCH = 20
N_TRIALS = 150
N_GPU_WORKERS = 2

PLOT_DIR = "optuna_plots_ep_fmist"


def get_visible_gpu_tokens():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() != "":
        tokens = [t.strip() for t in cuda_visible.split(",") if t.strip()]
        return tokens

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("GPU ")]
    return [str(i) for i in range(len(lines))]


def load_runtime_deps():
    import jax
    import jax.numpy as jnp
    import optax

    import torch
    import torchvision
    import torchvision.transforms as transforms

    import pcx as px
    import pcx.nn as pxnn
    import pcx.predictive_coding as pxc
    import pcx.utils as pxu

    from deq.models import FCMultiVodeDEQPCModel
    from deq.training import train_on_batch_ep, eval_on_batch_ep
    from deq.evaluation import evaluate_accuracy
    from deq.spectral_norm import _project_weight

    return {
        "jax": jax,
        "jnp": jnp,
        "optax": optax,
        "torch": torch,
        "torchvision": torchvision,
        "transforms": transforms,
        "px": px,
        "pxnn": pxnn,
        "pxc": pxc,
        "pxu": pxu,
        "evaluate_accuracy": evaluate_accuracy,
        "FCMultiVodeDEQPCModel": FCMultiVodeDEQPCModel,
        "train_on_batch_ep": train_on_batch_ep,
        "eval_on_batch_ep": eval_on_batch_ep,
        "_project_weight": _project_weight,
    }


class NoOpProjector:
    def project(self):
        return None


class FCSpectralProjector:
    """Spectral projector for FC implicit blocks (f1, f2)."""

    def __init__(self, model, project_weight, target: float = 0.95, jnp=None):
        self.target = target
        self._project_weight = project_weight
        self._jnp = jnp
        self._entries = []

        for block in [model.f1, model.f2]:
            wp = block.linear.nn.weight
            w = wp.get()
            fan_in = w.reshape(w.shape[0], -1).shape[1]
            v = self._jnp.ones((fan_in,)) / self._jnp.sqrt(fan_in)
            self._entries.append([wp, v])

    def project(self):
        for entry in self._entries:
            wp, v = entry
            w_new, v_new, _ = self._project_weight(wp.get(), v, self.target)
            wp.set(w_new)
            entry[1] = v_new


def get_fashion_mnist(train_bs, test_bs, *, root, torch, torchvision, transforms):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        train_ds = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            transform=tfm,
            download=True,
        )
        test_ds = torchvision.datasets.FashionMNIST(
            root=root,
            train=False,
            transform=tfm,
            download=True,
        )

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=test_bs,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    return train_dl, test_dl


def assert_gpu_backend(jax):
    backend = jax.default_backend()
    if backend != "gpu":
        devices = ", ".join(f"{d.platform}:{d.device_kind}" for d in jax.devices())
        print(
            f"GPU required but current JAX backend is '{backend}'. "
            f"Available devices: {devices}",
            file=sys.stderr,
        )
        sys.exit(1)


def train_epoch(train_dl, T_free, T_nudg, lr_decay_h, *,
                model, optim_w, optim_h, projector, deps):
    jax = deps["jax"]
    train_on_batch_ep = deps["train_on_batch_ep"]
    for x, y in train_dl:
        train_on_batch_ep(
            T_free,
            T_nudg,
            x.numpy(),
            jax.nn.one_hot(y.numpy(), N_CLASSES),
            lr_decay_h,
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
        )
        projector.project()


def make_objective(*, n_epochs: int, deps):
    jax = deps["jax"]
    jnp = deps["jnp"]
    optax = deps["optax"]
    torch = deps["torch"]
    torchvision = deps["torchvision"]
    transforms = deps["transforms"]
    px = deps["px"]
    pxnn = deps["pxnn"]
    pxc = deps["pxc"]
    pxu = deps["pxu"]
    evaluate_accuracy = deps["evaluate_accuracy"]
    eval_on_batch_ep = deps["eval_on_batch_ep"]
    FCMultiVodeDEQPCModel = deps["FCMultiVodeDEQPCModel"]
    project_weight = deps["_project_weight"]

    _dl_cache = {}

    def _get_dls(batch_size: int):
        if batch_size not in _dl_cache:
            _dl_cache[batch_size] = get_fashion_mnist(
                batch_size,
                TEST_BATCH_SIZE,
                root=DATA_ROOT,
                torch=torch,
                torchvision=torchvision,
                transforms=transforms,
            )
        return _dl_cache[batch_size]

    baseline = dict(
        batch_size=128,
        hidden_dim=64,
        T_free=20,
        T_nudg=40,
        nudging=0.01,
        init_scale=0.002,
        stop_grad_f=True,
        energy_type="se",
        lr_w=0.0004,
        wd_w=0.028,
        lr_h=0.8,
        mom_h=0.3,
        lr_decay_h=1.0,
        schedule_scale_1=0.8,
        schedule_scale_2=0.9,
        use_spectral_norm=False,
        spectral_constant=0.9,
    )

    def objective(trial: optuna.Trial) -> float:
        jax.clear_caches()
        px.RKG.seed(SEED)

        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        hidden_dim = trial.suggest_categorical("hidden_dim", HIDDEN_DIM_CHOICES)
        T_free = trial.suggest_categorical("T_free", T_FREE_CHOICES)
        T_nudg = trial.suggest_categorical("T_nudg", T_NUDG_CHOICES)

        nudging = trial.suggest_float("nudging", 0.005, 0.05, log=True)
        init_scale = trial.suggest_float("init_scale", 0.0005, 0.05, log=True)
        stop_grad_f = trial.suggest_categorical("stop_grad_f", [True])
        energy_type = trial.suggest_categorical("energy_type", ["se"])

        lr_w = trial.suggest_float("lr_w", 0.0001, 0.0005, log=True)
        wd_w = trial.suggest_float("wd_w", 0.01, 1.0, log=True)
        lr_h = trial.suggest_float("lr_h", 0.1, 1.0, log=True)
        mom_h = trial.suggest_float("mom_h", 0.2, 0.5, log=False)
        lr_decay_h = trial.suggest_float("lr_decay_h", 0.99, 1.0)

        schedule_scale_1 = trial.suggest_float("schedule_scale_1", 0.2, 1.0)
        schedule_scale_2 = trial.suggest_float("schedule_scale_2", 0.2, 1.0)

        use_spectral_norm = trial.suggest_categorical("use_spectral_norm", [False])
        spectral_constant = trial.suggest_float("spectral_constant", 0.90, 0.99)

        train_dl, test_dl = _get_dls(batch_size)

        model = FCMultiVodeDEQPCModel(
            hidden_dim=hidden_dim,
            input_dim=784,
            n_classes=N_CLASSES,
            nudging=nudging,
            init_scale=init_scale,
            stop_grad_f=stop_grad_f,
            energy_type=energy_type,
        )

        if use_spectral_norm:
            projector = FCSpectralProjector(
                model,
                project_weight=project_weight,
                target=spectral_constant,
                jnp=jnp,
            )
        else:
            projector = NoOpProjector()

        optim_h = pxu.Optim(
            lambda: optax.sgd(lr_h, momentum=mom_h, nesterov=True),
        )

        steps_per_epoch = len(train_dl)
        schedule_w = optax.piecewise_constant_schedule(
            init_value=lr_w,
            boundaries_and_scales={
                5 * steps_per_epoch: schedule_scale_1,
                10 * steps_per_epoch: schedule_scale_2,
            },
        )
        optim_w = pxu.Optim(
            lambda: optax.adamw(schedule_w, weight_decay=wd_w),
            pxu.M(pxnn.LayerParam)(model),
        )

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            x0 = jnp.zeros((batch_size, 1, 28, 28))
            y0 = jnp.zeros((batch_size, N_CLASSES))
            x_inj = jax.vmap(lambda xi: model.embed(xi))(x0)
            model.x_inj_cache.set(x_inj)
            for vode in model.get_latent_vodes():
                vode.h.set(jnp.zeros_like(x_inj))
            model.vode_out.h.set(y0)

        best = 0.0
        for epoch in range(1, n_epochs + 1):
            train_epoch(
                train_dl,
                T_free,
                T_nudg,
                lr_decay_h,
                model=model,
                optim_w=optim_w,
                optim_h=optim_h,
                projector=projector,
                deps=deps,
            )
            acc = evaluate_accuracy(
                test_dl,
                (T_free, T_nudg),
                model=model,
                optim_h=optim_h,
                eval_fn=eval_on_batch_ep,
                lr_decay=lr_decay_h,
            )

            best = max(best, acc)
            trial.report(acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best

    return objective, baseline


def run_worker(gpu_token: str, worker_rank: int, n_trials_worker: int,
               storage: str, study_name: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
    deps = load_runtime_deps()
    jax = deps["jax"]
    px = deps["px"]

    assert_gpu_backend(jax)
    px.RKG.seed(SEED + worker_rank)

    objective, baseline = make_objective(
        n_epochs=N_EPOCHS_SEARCH,
        deps=deps,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED + worker_rank, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    if worker_rank == 0 and len(study.trials) == 0:
        study.enqueue_trial(baseline)

    study.optimize(
        objective,
        n_trials=n_trials_worker,
        gc_after_trial=True,
        show_progress_bar=False,
    )


def save_study_plots(study: optuna.Study, out_dir: str):
    """Save key Optuna visualisations as PNGs."""
    import matplotlib
    matplotlib.use("Agg")

    from optuna.visualization.matplotlib import (
        plot_param_importances,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_slice,
    )
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    plots = {
        "param_importances": plot_param_importances,
        "optimization_history": plot_optimization_history,
        "parallel_coordinate": plot_parallel_coordinate,
        "slice": plot_slice,
    }

    for name, plot_fn in plots.items():
        try:
            fig = plot_fn(study)
            if isinstance(fig, matplotlib.figure.Figure):
                fig.savefig(
                    os.path.join(out_dir, f"{name}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
            else:
                fig.figure.savefig(
                    os.path.join(out_dir, f"{name}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
            plt.close("all")
            print(f"  Saved {name}.png")
        except Exception as e:
            print(f"  Skipped {name}: {e}")


def main():
    storage = "sqlite:///deqpc_optuna_fmist.db"
    study_name = f"ep_fmnist_{N_EPOCHS_SEARCH}_epochs"

    visible_gpus = get_visible_gpu_tokens()
    workers = min(N_GPU_WORKERS, len(visible_gpus))
    if workers == 0:
        print(
            "No CUDA GPUs are visible in this session. "
            "Check CUDA_VISIBLE_DEVICES and scheduler GPU allocation.",
            file=sys.stderr,
        )
        sys.exit(1)

    if workers < N_GPU_WORKERS:
        print(
            f"Requested {N_GPU_WORKERS} GPU workers, but only {workers} GPU(s) "
            f"are visible ({visible_gpus}). Running with {workers} worker(s)."
        )

    trials_per_worker = [N_TRIALS // workers] * workers
    for i in range(N_TRIALS % workers):
        trials_per_worker[i] += 1

    ctx = mp.get_context("spawn")
    procs = []
    for worker_rank, gpu_token in enumerate(visible_gpus[:workers]):
        p = ctx.Process(
            target=run_worker,
            args=(
                gpu_token,
                worker_rank,
                trials_per_worker[worker_rank],
                storage,
                study_name,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            exit_code = p.exitcode if p.exitcode is not None else 1

    if exit_code != 0:
        for p in procs:
            if p.is_alive():
                p.terminate()
        sys.exit(exit_code)

    study = optuna.load_study(study_name=study_name, storage=storage)
    print("\nBest value (accuracy):", study.best_value)
    print("Best params:", study.best_params)

    print(f"\nSaving plots to {PLOT_DIR}/")
    save_study_plots(study, PLOT_DIR)


if __name__ == "__main__":
    main()
