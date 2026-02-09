"""Analytical expressions for prior moments.

Used to verify sample statistics in tests.
"""

import jax.numpy as jnp

from trellis.prior import HalfNormal, LogNormal, NoPrior, Normal


def _normal_mean(prior, params):
    return jnp.asarray(prior.loc)


def _normal_var(prior, params):
    return jnp.asarray(prior.scale**2)


def _lognormal_mean(prior, params):
    return jnp.exp(prior.loc + prior.scale**2 / 2)


def _lognormal_var(prior, params):
    return (jnp.exp(prior.scale**2) - 1) * jnp.exp(
        2 * prior.loc + prior.scale**2
    )


def _halfnormal_mean(prior, params):
    return prior.scale * jnp.sqrt(2 / jnp.pi)


def _halfnormal_var(prior, params):
    return prior.scale**2 * (1 - 2 / jnp.pi)


def _noprior_mean(prior, params):
    return jnp.asarray(params.value)


def _noprior_var(prior, params):
    return jnp.zeros_like(jnp.asarray(params.value))


analytical = {
    Normal: {'mean': _normal_mean, 'var': _normal_var},
    LogNormal: {'mean': _lognormal_mean, 'var': _lognormal_var},
    HalfNormal: {'mean': _halfnormal_mean, 'var': _halfnormal_var},
    NoPrior: {'mean': _noprior_mean, 'var': _noprior_var},
}
