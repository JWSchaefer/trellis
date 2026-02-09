import pytest
from beartype.typing import Tuple, Type
from jax import Array
from jax import numpy as jnp
from jax.random import PRNGKey

from trellis import Transform
from trellis.prior import Prior

from .analytical import analytical
from .shared import (
    prior_id,
    prior_inits,
    shape_id,
    shapes,
    transform_id,
    transforms,
    values,
    values_id,
)


@pytest.mark.parametrize('prior_init', prior_inits, ids=prior_id)
@pytest.mark.parametrize('transform', transforms, ids=transform_id)
@pytest.mark.parametrize('value', values, ids=values_id)
@pytest.mark.parametrize('shape', shapes, ids=shape_id)
def test_sample_shape(
    prior_init: Tuple[Type[Prior], dict[str, float]],
    transform: Type[Transform],
    value: float | Array,
    shape: Tuple[int, ...],
):
    prior_type, args = prior_init
    prior = prior_type(transform=transform, value=value, **args)
    params = prior.init_params()
    state = prior.init_state()

    key = PRNGKey(42)

    samples = prior.sample(key, params, state, shape=shape)
    if isinstance(value, Array):
        assert samples.shape[-1] == value.shape[0]
        assert samples.shape[:-1] == shape
    else:
        assert samples.shape == shape


@pytest.mark.parametrize('prior_init', prior_inits, ids=prior_id)
@pytest.mark.parametrize('transform', transforms, ids=transform_id)
def test_sample_mean(
    prior_init: Tuple[Type[Prior], dict[str, float]],
    transform: Type[Transform],
):
    n_samples: int = 100_000
    n_std_errors: float = 4.0  # ~99.99% pass rate

    prior_type, args = prior_init

    prior: Prior = prior_type(transform=transform, value=0.0, **args)
    params = prior.init_params()
    state = prior.init_state()

    key = PRNGKey(42)

    samples = prior.sample(key, params, state, shape=(n_samples,))

    analytical_mean = analytical[prior_type]['mean'](prior, params)
    analytical_var = analytical[prior_type]['var'](prior, params)

    statistical_mean = jnp.mean(samples)
    statistical_var = jnp.var(samples)

    # Tolerance based on standard error of the mean
    mean_std_error = jnp.sqrt(analytical_var / n_samples)
    mean_atol = n_std_errors * mean_std_error

    # Tolerance based on standard error of the variance
    var_std_error = analytical_var * jnp.sqrt(2 / n_samples)
    var_atol = n_std_errors * var_std_error

    assert jnp.allclose(
        statistical_mean, analytical_mean, atol=mean_atol, rtol=0
    ), (
        f'Mean mismatch: got {statistical_mean}, expected {analytical_mean} '
        f'(atol={mean_atol})'
    )
    assert jnp.allclose(
        statistical_var, analytical_var, atol=var_atol, rtol=0
    ), (
        f'Var mismatch: got {statistical_var}, expected {analytical_var} '
        f'(atol={var_atol})'
    )
