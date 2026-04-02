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

    @requires(state=(SampleSpec, 'computed'))
    def use_computed(state: 'SampleSpec.CheckedState') -> Array:
        return state.computed * 2

    spec = SampleSpec()
    state = spec.init_state()
    state = replace(state, computed=jnp.array([1.0]))

    result = use_computed(state)
    assert jnp.allclose(result, jnp.array([2.0]))


def test_requires_decorator_failure():
    """@requires decorator raises when fields are missing."""

    @requires(state=(SampleSpec, 'computed'))
    def use_computed(state: 'SampleSpec.CheckedState') -> Array:
        return state.computed * 2

    spec = SampleSpec()
    state = spec.init_state()  # All None

    with pytest.raises(StateValidationError):
        use_computed(state)


def test_requires_decorator_with_kwargs():
    """@requires decorator works with keyword arguments."""

    @requires(state=(SampleSpec, 'computed'))
    def use_computed(state: 'SampleSpec.CheckedState', multiplier: float) -> Array:
        return state.computed * multiplier

    spec = SampleSpec()
    state = spec.init_state()
    state = replace(state, computed=jnp.array([1.0]))

    result = use_computed(state=state, multiplier=3.0)
    assert jnp.allclose(result, jnp.array([3.0]))


def test_requires_decorator_preserves_metadata():
    """@requires decorator preserves function metadata."""

    @requires(state=(SampleSpec, 'computed'))
    def my_function(state: 'SampleSpec.CheckedState') -> Array:
        """Docstring for my_function."""
        return state.computed

    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Docstring for my_function.'


def test_requires_string_forward_reference():
    """@requires works with string class name (forward reference)."""

    @requires(state=('SampleSpec', 'computed'))
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

    @requires(state=('SpecWithMethod', 'value'))
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


# Multi-state tests


class AnotherSpec(Spec):
    """Second Spec class for testing multi-state requirements."""

    weight_m: State[Array]
    weight_k: State[Array]


def test_requires_multiple_states():
    """New API validates multiple state parameters."""

    @requires(
        state=(SampleSpec, 'computed'),
        other_state=(AnotherSpec, 'weight_m'),
    )
    def combine(
        state: 'SampleSpec.CheckedState',
        other_state: 'AnotherSpec.CheckedState',
    ) -> Array:
        return state.computed + other_state.weight_m

    sample_spec = SampleSpec()
    sample_state = sample_spec.init_state()
    sample_state = replace(sample_state, computed=jnp.array([1.0]))

    another_spec = AnotherSpec()
    another_state = another_spec.init_state()
    another_state = replace(another_state, weight_m=jnp.array([2.0]))

    result = combine(sample_state, another_state)
    assert jnp.allclose(result, jnp.array([3.0]))


def test_requires_multiple_states_with_kwargs():
    """New API works with keyword arguments."""

    @requires(
        state=(SampleSpec, 'computed'),
        other_state=(AnotherSpec, 'weight_m'),
    )
    def combine(
        state: 'SampleSpec.CheckedState',
        other_state: 'AnotherSpec.CheckedState',
    ) -> Array:
        return state.computed + other_state.weight_m

    sample_spec = SampleSpec()
    sample_state = sample_spec.init_state()
    sample_state = replace(sample_state, computed=jnp.array([1.0]))

    another_spec = AnotherSpec()
    another_state = another_spec.init_state()
    another_state = replace(another_state, weight_m=jnp.array([2.0]))

    result = combine(state=sample_state, other_state=another_state)
    assert jnp.allclose(result, jnp.array([3.0]))


def test_requires_multiple_states_failure_first():
    """Error on first state reports correct parameter."""

    @requires(
        state=(SampleSpec, 'computed'),
        other_state=(AnotherSpec, 'weight_m'),
    )
    def combine(
        state: 'SampleSpec.CheckedState',
        other_state: 'AnotherSpec.CheckedState',
    ) -> Array:
        return state.computed + other_state.weight_m

    sample_spec = SampleSpec()
    sample_state = sample_spec.init_state()  # computed is None

    another_spec = AnotherSpec()
    another_state = another_spec.init_state()
    another_state = replace(another_state, weight_m=jnp.array([2.0]))

    with pytest.raises(StateValidationError) as exc:
        combine(sample_state, another_state)

    assert exc.value.param_name == 'state'
    assert 'computed' in str(exc.value)


def test_requires_multiple_states_failure_second():
    """Error on second state reports correct parameter."""

    @requires(
        state=(SampleSpec, 'computed'),
        other_state=(AnotherSpec, 'weight_m'),
    )
    def combine(
        state: 'SampleSpec.CheckedState',
        other_state: 'AnotherSpec.CheckedState',
    ) -> Array:
        return state.computed + other_state.weight_m

    sample_spec = SampleSpec()
    sample_state = sample_spec.init_state()
    sample_state = replace(sample_state, computed=jnp.array([1.0]))

    another_spec = AnotherSpec()
    another_state = another_spec.init_state()  # weight_m is None

    with pytest.raises(StateValidationError) as exc:
        combine(sample_state, another_state)

    assert exc.value.param_name == 'other_state'
    assert 'weight_m' in str(exc.value)


def test_requires_multiple_fields_per_state():
    """Each state can require multiple fields."""

    @requires(
        state=(SampleSpec, 'computed', 'cached'),
        other_state=(AnotherSpec, 'weight_m', 'weight_k'),
    )
    def combine(
        state: 'SampleSpec.CheckedState',
        other_state: 'AnotherSpec.CheckedState',
    ) -> Array:
        return state.computed + state.cached + other_state.weight_m + other_state.weight_k

    sample_spec = SampleSpec()
    sample_state = sample_spec.init_state()
    sample_state = replace(
        sample_state,
        computed=jnp.array([1.0]),
        cached=jnp.array([2.0]),
    )

    another_spec = AnotherSpec()
    another_state = another_spec.init_state()
    another_state = replace(
        another_state,
        weight_m=jnp.array([3.0]),
        weight_k=jnp.array([4.0]),
    )

    result = combine(sample_state, another_state)
    assert jnp.allclose(result, jnp.array([10.0]))


def test_requires_string_forward_reference_multiple():
    """New API works with string forward references."""

    @requires(
        state=('SampleSpec', 'computed'),
        other_state=('AnotherSpec', 'weight_m'),
    )
    def combine(
        state: 'SampleSpec.CheckedState',
        other_state: 'AnotherSpec.CheckedState',
    ) -> Array:
        return state.computed + other_state.weight_m

    sample_spec = SampleSpec()
    sample_state = sample_spec.init_state()
    sample_state = replace(sample_state, computed=jnp.array([1.0]))

    another_spec = AnotherSpec()
    another_state = another_spec.init_state()
    another_state = replace(another_state, weight_m=jnp.array([2.0]))

    result = combine(sample_state, another_state)
    assert jnp.allclose(result, jnp.array([3.0]))


class SpecWithMultiStateMethod(Spec):
    """Spec with a method that validates multiple states."""

    value: State[Array]

    @requires(
        state=('SpecWithMultiStateMethod', 'value'),
        other_state=('AnotherSpec', 'weight_m'),
    )
    def combine(
        self,
        params: 'SpecWithMultiStateMethod.Params',
        state: 'SpecWithMultiStateMethod.CheckedState',
        other_state: 'AnotherSpec.CheckedState',
    ) -> Array:
        return state.value + other_state.weight_m


def test_requires_method_with_multiple_states():
    """@requires works on methods with multiple state parameters."""
    spec = SpecWithMultiStateMethod()
    params = spec.init_params()
    state = spec.init_state()
    state = replace(state, value=jnp.array([2.0]))

    another_spec = AnotherSpec()
    another_state = another_spec.init_state()
    another_state = replace(another_state, weight_m=jnp.array([3.0]))

    result = spec.combine(params, state, another_state)
    assert jnp.allclose(result, jnp.array([5.0]))


def test_requires_empty_raises():
    """Calling requires() with no arguments raises an error."""
    with pytest.raises(ValueError, match="needs at least one"):

        @requires()
        def bad_func(state):
            pass


def test_requires_param_not_found_raises():
    """Unknown parameter name raises an error."""

    @requires(nonexistent_param=(SampleSpec, 'computed'))
    def bad_func(state):
        pass

    spec = SampleSpec()
    state = spec.init_state()
    state = replace(state, computed=jnp.array([1.0]))

    with pytest.raises(ValueError, match="not found"):
        bad_func(state)


def test_requires_no_fields_still_converts():
    """New API with no field requirements still converts to CheckedState."""

    @requires(state=(SampleSpec,))
    def use_state(state: 'SampleSpec.CheckedState') -> bool:
        # Just check it's a CheckedState (has all fields as non-optional)
        return True

    spec = SampleSpec()
    state = spec.init_state()
    # No fields required, so even empty state should work
    result = use_state(state)
    assert result is True
