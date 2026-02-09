"""Example: Sampling from priors."""

import jax.numpy as jnp
import jax.random as jr

from trellis.prior import HalfNormal, LogNormal, Normal, NoPrior


key = jr.PRNGKey(42)

# Normal prior
print('Normal prior (loc=0, scale=1):')
normal = Normal(value=0.0, loc=0.0, scale=1.0)
normal_params = normal.init_params()
normal_state = normal.init_state()

samples = normal.sample(key, normal_params, normal_state, shape=(5,))
print(f'  samples: {samples}')
print(f'  log_prob: {normal.log_prob(samples[0], normal_params, normal_state)}')

# LogNormal prior (positive values)
print('\nLogNormal prior (loc=0, scale=1):')
lognormal = LogNormal(value=1.0, loc=0.0, scale=1.0)
ln_params = lognormal.init_params()
ln_state = lognormal.init_state()

key, subkey = jr.split(key)
samples = lognormal.sample(subkey, ln_params, ln_state, shape=(5,))
print(f'  samples: {samples}')
print(f'  all positive: {jnp.all(samples > 0)}')

# HalfNormal prior (positive values)
print('\nHalfNormal prior (scale=1):')
halfnormal = HalfNormal(value=1.0, scale=1.0)
hn_params = halfnormal.init_params()
hn_state = halfnormal.init_state()

key, subkey = jr.split(key)
samples = halfnormal.sample(subkey, hn_params, hn_state, shape=(5,))
print(f'  samples: {samples}')
print(f'  all positive: {jnp.all(samples > 0)}')

# NoPrior (deterministic, no distribution)
print('\nNoPrior (deterministic):')
noprior = NoPrior(value=42.0)
np_params = noprior.init_params()
np_state = noprior.init_state()

key, subkey = jr.split(key)
samples = noprior.sample(subkey, np_params, np_state, shape=(5,))
print(f'  samples: {samples}')  # Always returns the value
print(f'  log_prob: {noprior.log_prob(jnp.array(42.0), np_params, np_state)}')  # Always 0
