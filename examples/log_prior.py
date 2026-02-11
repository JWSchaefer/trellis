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
        loc=Normal(value=0.0, loc=0.0, scale=10.0),
        scale=HalfNormal(value=1.0, scale=1.0),
    )
)
learnable_params = learnable_spec.init_params()

# Create a model with learnable hyperparameters
model = Model.from_spec(learnable_spec)

print(f'Model.params: {model.params}')
print(f'Model.params.ell.value: {model.params.ell.value}')
print(f'Model.params.ell.loc.value: {model.params.ell.loc.value}')
print(f'Model.params.ell.scale.value: {model.params.ell.scale.value}')

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

# 1. LogNormalLearnable.log_prob(ell) using loc and scale from params
ell_prior = learnable_spec.ell
ell_log_prob = ell_prior.log_prob(
    jnp.asarray(constrained_ell), learnable_params.ell, ell_prior.init_state()
)
print(f'  LogNormalLearnable.log_prob(ell) = {ell_log_prob}')

# 2. Hyperprior on loc: Normal.log_prob(loc)
loc_prior = learnable_spec.ell.loc
loc_val = learnable_params.ell.loc.value
loc_params = loc_prior.init_params()
loc_state = loc_prior.init_state()
loc_log_prob = loc_prior.log_prob(jnp.asarray(loc_val), loc_params, loc_state)
print(f'  Normal.log_prob(loc={loc_val}) = {loc_log_prob}')

# 3. Hyperprior on scale: HalfNormal.log_prob(scale)
scale_prior = learnable_spec.ell.scale
scale_val = learnable_params.ell.scale.value
scale_params = scale_prior.init_params()
scale_state = scale_prior.init_state()
scale_log_prob = scale_prior.log_prob(
    jnp.asarray(scale_val), scale_params, scale_state
)
print(f'  HalfNormal.log_prob(scale={scale_val}) = {scale_log_prob}')

# 4. Jacobian corrections
log_transform = Log()
jacobian_ell = log_transform.log_det_jacobian(unconstrained.ell.value)
jacobian_scale = log_transform.log_det_jacobian(unconstrained.ell.scale.value)
print(f'  Jacobian (ell Log transform) = {jacobian_ell}')
print(f'  Jacobian (scale Log transform) = {jacobian_scale}')

manual_total = (
    ell_log_prob
    + loc_log_prob
    + scale_log_prob
    + jacobian_ell
    + jacobian_scale
)
print(f'\n  Manual total = {manual_total}')
print(f'  Model.log_prior = {log_prior_value}')
print(f'  Match: {jnp.allclose(manual_total, log_prior_value)}')
