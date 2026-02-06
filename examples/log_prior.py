"""Example: Model.log_prior() with hyperprior evaluation."""

import jax.numpy as jnp

from trellis import Model, Spec
from trellis.prior import HalfNormal, LogNormalLearnable, Normal
from trellis.transform import Log


class LearnablePriorSpec(Spec):
    ell: LogNormalLearnable[float, Normal[float], HalfNormal[float]]


learnable_spec = LearnablePriorSpec(
    ell=LogNormalLearnable(
        value=1.0,
        mu=Normal(value=0.0, loc=0.0, scale=10.0),
        sigma=HalfNormal(value=1.0, scale=1.0),
    )
)
learnable_params = learnable_spec.init_params()

# Create a model with learnable hyperparameters
model = Model.from_spec(learnable_spec)

print(f'Model.params: {model.params}')
print(f'Model.params.ell.value: {model.params.ell.value}')
print(f'Model.params.ell.mu.value: {model.params.ell.mu.value}')
print(f'Model.params.ell.sigma.value: {model.params.ell.sigma.value}')

# Get unconstrained params
unconstrained = model.to_unconstrained()
print(f'\nUnconstrained params: {unconstrained}')

# Evaluate log prior (includes hyperpriors!)
log_prior_value = model.log_prior(unconstrained)
print(f'\nlog_prior (with hyperpriors): {log_prior_value}')

# Manual verification
constrained_ell = model.params.ell.value
print('\nManual verification:')
print(f'  constrained ell = {constrained_ell}')

# 1. LogNormalLearnable.log_prob(ell) using mu and sigma from params
ell_prior = learnable_spec.ell
ell_log_prob = ell_prior.log_prob(
    jnp.asarray(constrained_ell), learnable_params.ell, ell_prior.init_state()
)
print(f'  LogNormalLearnable.log_prob(ell) = {ell_log_prob}')

# 2. Hyperprior on mu: Normal.log_prob(mu)
mu_prior = learnable_spec.ell.mu
mu_val = learnable_params.ell.mu.value
mu_params = mu_prior.init_params()
mu_state = mu_prior.init_state()
mu_log_prob = mu_prior.log_prob(jnp.asarray(mu_val), mu_params, mu_state)
print(f'  Normal.log_prob(mu={mu_val}) = {mu_log_prob}')

# 3. Hyperprior on sigma: HalfNormal.log_prob(sigma)
sigma_prior = learnable_spec.ell.sigma
sigma_val = learnable_params.ell.sigma.value
sigma_params = sigma_prior.init_params()
sigma_state = sigma_prior.init_state()
sigma_log_prob = sigma_prior.log_prob(
    jnp.asarray(sigma_val), sigma_params, sigma_state
)
print(f'  HalfNormal.log_prob(sigma={sigma_val}) = {sigma_log_prob}')

# 4. Jacobian corrections
log_transform = Log()
jacobian_ell = log_transform.log_det_jacobian(unconstrained.ell.value)
jacobian_sigma = log_transform.log_det_jacobian(unconstrained.ell.sigma.value)
print(f'  Jacobian (ell Log transform) = {jacobian_ell}')
print(f'  Jacobian (sigma Log transform) = {jacobian_sigma}')

manual_total = (
    ell_log_prob + mu_log_prob + sigma_log_prob + jacobian_ell + jacobian_sigma
)
print(f'\n  Manual total = {manual_total}')
print(f'  Model.log_prior = {log_prior_value}')
print(f'  Match: {jnp.allclose(manual_total, log_prior_value)}')
