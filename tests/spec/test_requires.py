from dataclasses import replace

import jax.numpy as jnp
import pytest
from jax import Array

from trellis import Spec, State
from trellis.spec import StateValidationError, requires


class SampleSpec(Spec):
    computed: State[Array]
    cached: State[Array]


def test_checked_state_generated():
    """CheckedState class is generated for Spec with State fields."""
    assert hasattr(SampleSpec, 'CheckedState')
    checked_cls = SampleSpec.CheckedState
    assert checked_cls.__name__ == 'SampleSpecCheckedState'


def test_check_state_success():
    """check_state returns CheckedState when required fields are present."""
    spec = SampleSpec()
    state = spec.init_state()
    state = replace(state, computed=jnp.array([1.0]), cached=jnp.array([2.0]))

    checked = SampleSpec.check_state(state, 'computed')
    assert checked.computed is not None
    assert jnp.allclose(checked.computed, jnp.array([1.0]))


def test_check_state_multiple_fields():
    """check_state validates multiple fields."""
    spec = SampleSpec()
    state = spec.init_state()
    state = replace(state, computed=jnp.array([1.0]), cached=jnp.array([2.0]))

    checked = SampleSpec.check_state(state, 'computed', 'cached')
    assert checked.computed is not None
    assert checked.cached is not None


def test_check_state_failure():
    """check_state raises StateValidationError when fields are None."""
    spec = SampleSpec()
    state = spec.init_state()  # All None

    with pytest.raises(StateValidationError) as exc:
        SampleSpec.check_state(state, 'computed')

    assert 'computed' in str(exc.value)
    assert exc.value.spec_name == 'SampleSpec'
    assert exc.value.missing_fields == ['computed']


def test_check_state_failure_multiple():
    """check_state reports all missing fields."""
    spec = SampleSpec()
    state = spec.init_state()  # All None

    with pytest.raises(StateValidationError) as exc:
        SampleSpec.check_state(state, 'computed', 'cached')

    assert 'computed' in str(exc.value)
    assert 'cached' in str(exc.value)
    assert set(exc.value.missing_fields) == {'computed', 'cached'}


def test_requires_decorator():
    """@requires decorator validates state and converts to CheckedState."""

    @requires(SampleSpec, 'computed')
    def use_computed(state: 'SampleSpec.CheckedState') -> Array:
        return state.computed * 2

    spec = SampleSpec()
    state = spec.init_state()
    state = replace(state, computed=jnp.array([1.0]))

    result = use_computed(state)
    assert jnp.allclose(result, jnp.array([2.0]))


def test_requires_decorator_failure():
    """@requires decorator raises when fields are missing."""

    @requires(SampleSpec, 'computed')
    def use_computed(state: 'SampleSpec.CheckedState') -> Array:
        return state.computed * 2

    spec = SampleSpec()
    state = spec.init_state()  # All None

    with pytest.raises(StateValidationError):
        use_computed(state)


def test_requires_decorator_with_kwargs():
    """@requires decorator works with keyword arguments."""

    @requires(SampleSpec, 'computed')
    def use_computed(state: 'SampleSpec.CheckedState', multiplier: float) -> Array:
        return state.computed * multiplier

    spec = SampleSpec()
    state = spec.init_state()
    state = replace(state, computed=jnp.array([1.0]))

    result = use_computed(state=state, multiplier=3.0)
    assert jnp.allclose(result, jnp.array([3.0]))


def test_requires_decorator_preserves_metadata():
    """@requires decorator preserves function metadata."""

    @requires(SampleSpec, 'computed')
    def my_function(state: 'SampleSpec.CheckedState') -> Array:
        """Docstring for my_function."""
        return state.computed

    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Docstring for my_function.'


def test_requires_string_forward_reference():
    """@requires works with string class name (forward reference)."""

    @requires('SampleSpec', 'computed')
    def use_computed(state: 'SampleSpec.CheckedState') -> Array:
        return state.computed * 2

    spec = SampleSpec()
    state = spec.init_state()
    state = replace(state, computed=jnp.array([1.0]))

    result = use_computed(state)
    assert jnp.allclose(result, jnp.array([2.0]))


class SpecWithMethod(Spec):
    """Spec with a method that uses @requires with forward reference."""

    value: State[Array]

    @requires('SpecWithMethod', 'value')
    def compute(
        self,
        params: 'SpecWithMethod.Params',
        state: 'SpecWithMethod.CheckedState',
    ) -> Array:
        return state.value * 3


def test_requires_method_with_forward_reference():
    """@requires works on methods with string forward reference."""
    spec = SpecWithMethod()
    params = spec.init_params()
    state = spec.init_state()
    state = replace(state, value=jnp.array([2.0]))

    result = spec.compute(params, state)
    assert jnp.allclose(result, jnp.array([6.0]))


def test_requires_method_failure():
    """@requires on methods raises when fields are missing."""
    spec = SpecWithMethod()
    params = spec.init_params()
    state = spec.init_state()  # value is None

    with pytest.raises(StateValidationError) as exc:
        spec.compute(params, state)

    assert 'value' in str(exc.value)
