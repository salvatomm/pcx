"""Optuna hyperparameter search for the 2-vode MultiVodeDEQPCModel.

Spawns N_GPU_WORKERS processes, one per GPU, sharing a single SQLite study.
Batch size is searched over {64, 128, 256}.
After all workers finish, saves study visualisations to ``optuna_plots/``.
"""

import sys
import os
import multiprocessing as mp
import subprocess

import optuna

SEED = 0
N_CLASSES = 10
N_CHANNELS = 256

BATCH_SIZE_CHOICES = [64, 128, 256]
TEST_BATCH_SIZE = 1000
DATA_ROOT = "~/tmp/cifar10/"

N_EPOCHS_SEARCH = 20
N_TRIALS = 50
N_GPU_WORKERS = 2
STOP_GRAD_F = False
ENERGY_TYPE = "se"
LR_DECAY_H = 0.97
SPECTRAL_TARGET = 0.95

PLOT_DIR = "optuna_plots"


def get_visible_gpu_tokens():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() != "":
        tokens = [token.strip() for token in cuda_visible.split(",") if token.strip() != ""]
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

    import pcx as px
    import pcx.nn as pxnn
    import pcx.predictive_coding as pxc
    import pcx.utils as pxu

    from deq.models import MultiVodeDEQPCModel
    from deq.training import train_on_batch, eval_on_batch
    from deq.evaluation import evaluate_accuracy
    from deq.data import get_dataloaders
    from deq.spectral_norm import SpectralProjector

    return {
        "jax": jax,
        "jnp": jnp,
        "optax": optax,
        "px": px,
        "pxnn": pxnn,
        "pxc": pxc,
        "pxu": pxu,
        "evaluate_accuracy": evaluate_accuracy,
        "MultiVodeDEQPCModel": MultiVodeDEQPCModel,
        "get_dataloaders": get_dataloaders,
        "train_on_batch": train_on_batch,
        "eval_on_batch": eval_on_batch,
        "SpectralProjector": SpectralProjector,
    }


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


def train_epoch(train_dl, T_steps, lr_decay, *,
                model, optim_w, optim_h, projector, deps):
    jax = deps["jax"]
    train_on_batch = deps["train_on_batch"]
    for x, y in train_dl:
        train_on_batch(
            T_steps,
            x.numpy(),
            jax.nn.one_hot(y.numpy(), N_CLASSES),
            lr_decay,
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
        )
        projector.project()


def make_objective(*, n_epochs: int, deps, chan: int = N_CHANNELS):
    jax = deps["jax"]
    jnp = deps["jnp"]
    optax = deps["optax"]
    px = deps["px"]
    pxnn = deps["pxnn"]
    pxc = deps["pxc"]
    pxu = deps["pxu"]
    evaluate_accuracy = deps["evaluate_accuracy"]
    eval_on_batch = deps["eval_on_batch"]
    MultiVodeDEQPCModel = deps["MultiVodeDEQPCModel"]
    get_dataloaders = deps["get_dataloaders"]
    SpectralProjector = deps["SpectralProjector"]

    # Cache dataloaders per batch size to avoid re-downloading per trial.
    _dl_cache = {}

    def _get_dls(batch_size: int):
        if batch_size not in _dl_cache:
            _dl_cache[batch_size] = get_dataloaders(
                batch_size, TEST_BATCH_SIZE, root=DATA_ROOT, augmentation=True,
            )
        return _dl_cache[batch_size]

    baseline = dict(
        batch_size=256,
        T_train=120,
        nudging=0.17,
        lr_w=0.0008,
        wd_w=0.005,
        lr_h=0.15,
        mom_h=0.6,
        init_scale=0.005,
    )

    def objective(trial: optuna.Trial) -> float:
        px.RKG.seed(SEED)

        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        T_train    = trial.suggest_int("T_train", 60, 150)
        nudging    = trial.suggest_float("nudging", 0.01, 0.2, log=True)
        lr_w       = trial.suggest_float("lr_w", 0.00005, 0.001, log=True)
        wd_w       = trial.suggest_float("wd_w", 0.0001, 0.05, log=True)
        lr_h       = trial.suggest_float("lr_h", 0.1, 0.5, log=True)
        mom_h      = trial.suggest_float("mom_h", 0.3, 0.9, log=True)
        init_scale = trial.suggest_float("init_scale", 0.0001, 0.005, log=True)
        spectral_constant = trial.suggest_float("spectral_constant", 0.90, 0.99)

        T_eval = T_train
        train_dl, test_dl = _get_dls(batch_size)

        model = MultiVodeDEQPCModel(
            n_channels=chan,
            n_classes=N_CLASSES,
            nudging=nudging,
            init_scale=init_scale,
            stop_grad_f=STOP_GRAD_F,
            energy_type=ENERGY_TYPE,
        )

        projector = SpectralProjector(model, target=spectral_constant)

        optim_h = pxu.Optim(
            lambda: optax.sgd(lr_h, momentum=mom_h, nesterov=True),
        )

        steps_per_epoch = len(train_dl)
        schedule_w = optax.piecewise_constant_schedule(
            init_value=lr_w,
            boundaries_and_scales={
                20 * steps_per_epoch: 0.2,
                40 * steps_per_epoch: 0.2,
            },
        )
        optim_w = pxu.Optim(
            lambda: optax.adamw(schedule_w, weight_decay=wd_w),
            pxu.M(pxnn.LayerParam)(model),
        )

        # Warmup — model-agnostic interface (matches run_multi_vode.py)
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            x0 = jnp.zeros((batch_size, 3, 32, 32))
            y0 = jnp.zeros((batch_size, N_CLASSES))
            x_inj = jax.vmap(lambda xi: model.embed(xi))(x0)
            model.x_inj_cache.set(x_inj)
            for vode in model.get_latent_vodes():
                vode.h.set(jnp.zeros_like(x_inj))
            model.vode_out.h.set(y0)

        best = 0.0
        for epoch in range(1, n_epochs + 1):
            train_epoch(
                train_dl, T_train, LR_DECAY_H,
                model=model, optim_w=optim_w, optim_h=optim_h,
                projector=projector, deps=deps,
            )
            acc = evaluate_accuracy(
                test_dl, T_eval,
                model=model, optim_h=optim_h,
                eval_fn=eval_on_batch,
                lr_decay=LR_DECAY_H,
            )

            best = max(best, acc)
            trial.report(acc, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best

    return objective, baseline


def run_worker(gpu_token: str, worker_rank: int, n_trials_worker: int, storage: str, study_name: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
    deps = load_runtime_deps()
    jax = deps["jax"]
    px = deps["px"]

    assert_gpu_backend(jax)
    px.RKG.seed(SEED + worker_rank)

    objective, baseline = make_objective(
        n_epochs=N_EPOCHS_SEARCH,
        deps=deps,
        chan=N_CHANNELS,
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
        objective, n_trials=n_trials_worker,
        gc_after_trial=True, show_progress_bar=False,
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
                fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
            else:
                fig.figure.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"  Saved {name}.png")
        except Exception as e:
            print(f"  Skipped {name}: {e}")


def main():
    storage = "sqlite:///deqpc_optuna.db"
    study_name = f"multi_vode_{ENERGY_TYPE}_{N_EPOCHS_SEARCH}_epochs"

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
            f"Requested {N_GPU_WORKERS} GPU workers, but only {workers} GPU(s) are visible "
            f"({visible_gpus}). Running with {workers} worker(s)."
        )

    trials_per_worker = [N_TRIALS // workers] * workers
    for i in range(N_TRIALS % workers):
        trials_per_worker[i] += 1

    ctx = mp.get_context("spawn")
    procs = []
    for worker_rank, gpu_token in enumerate(visible_gpus[:workers]):
        p = ctx.Process(
            target=run_worker,
            args=(gpu_token, worker_rank, trials_per_worker[worker_rank], storage, study_name),
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