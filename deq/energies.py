"""Energy functions for predictive coding Vodes.

Two families
------------
- **Vode energy fns** — signature ``(vode, rkg) -> Array``, passed to
  ``pxc.Vode(energy_fn)`` and used by ``model.energy()`` for weight grads.
- **Diff energy fns** — signature ``(diff) -> scalar``, used inside the
  manual inference loop where we compute per-sample energy from raw diffs.

Both families must stay in sync: the Vode energy defines the loss landscape
for *weight* updates, while the diff energy defines the landscape for
*latent* updates.
"""

import jax
import jax.numpy as jnp
import pcx as px


# ======================================================================
#  Diff → scalar  (used inside inference loops)
# ======================================================================

def se_diff(diff: jax.Array) -> jax.Array:
    """Squared-error: 0.5 * ||diff||²."""
    return (0.5 * diff * diff).sum()


def log_cosh_diff(diff: jax.Array) -> jax.Array:
    """Log-cosh: Σ log(cosh(diff)).

    Numerically stable form: |d| + softplus(-2|d|) - log(2).
    Behaves like 0.5*d² for small d and |d| for large d.
    """
    a = jnp.abs(diff)
    return (a + jax.nn.softplus(-2.0 * a) - jnp.log(2.0)).sum()


DIFF_ENERGY_REGISTRY = {
    "se": se_diff,
    "log_cosh": log_cosh_diff,
}


# ======================================================================
#  Vode energy fns  (used by model.energy() → weight gradients)
# ======================================================================

def _se_vode_energy(vode, rkg=px.RKG):
    e = vode.get("h") - vode.get("u")
    return 0.5 * (e * e)


def _log_cosh_vode_energy(vode, rkg=px.RKG):
    e = vode.get("h") - vode.get("u")
    a = jnp.abs(e)
    return a + jax.nn.softplus(-2.0 * a) - jnp.log(2.0)


VODE_ENERGY_REGISTRY = {
    "se": _se_vode_energy,
    "log_cosh": _log_cosh_vode_energy,
}


def nudged_ce_energy(nudging: float):
    """Cross-entropy energy scaled by *nudging*, used on the output Vode."""
    def energy_fn(vode, rkg=px.RKG):
        return nudging * (-(vode.get("h") * jax.nn.log_softmax(vode.get("u"))))
    return energy_fn


# ======================================================================
#  Convenience
# ======================================================================

def get_vode_energy(energy_type: str):
    """Return the Vode-compatible energy fn for the given type string."""
    if energy_type not in VODE_ENERGY_REGISTRY:
        raise ValueError(f"Unknown energy type '{energy_type}'. "
                         f"Choose from {list(VODE_ENERGY_REGISTRY)}")
    return VODE_ENERGY_REGISTRY[energy_type]


def get_diff_energy(energy_type: str):
    """Return the diff→scalar energy fn for the given type string."""
    if energy_type not in DIFF_ENERGY_REGISTRY:
        raise ValueError(f"Unknown energy type '{energy_type}'. "
                         f"Choose from {list(DIFF_ENERGY_REGISTRY)}")
    return DIFF_ENERGY_REGISTRY[energy_type]
