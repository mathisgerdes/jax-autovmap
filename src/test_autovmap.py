import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_autovmap import autovmap


@pytest.mark.parametrize("args,kwargs", [
    ((2, 0), {}),
    ((2, 0, None), {}),
    ((), {'x': 2, 'y': 0}),
    ((), {'x': 2, 'y': 0, 'val': None}),
])
def test_basic_rank_specifications(args, kwargs):
    """Test different ways of specifying argument ranks."""
    def foo(x, y, val=1):
        assert jnp.ndim(x) == 2
        assert jnp.ndim(y) == 0
        return val

    vmapped = autovmap(*args, **kwargs)(foo)

    # no vmap needed
    result = vmapped(jnp.zeros((3, 3)), 0.0)
    assert result == 1

    # vmap second arg
    result = vmapped(jnp.zeros((3, 3)), jnp.array([0.0, 1.0, 2.0]))
    npt.assert_array_equal(result, np.ones([3]))

    # vmap first arg
    result = vmapped(jnp.zeros((5, 3, 3)), 0)
    npt.assert_array_equal(result, np.ones([5]))

    result = vmapped(jnp.zeros((5, 3, 3)), 0, val=0)
    npt.assert_array_equal(result, np.zeros([5]))

    result = vmapped(jnp.zeros((5, 3, 3)), 0, 0)
    npt.assert_array_equal(result, np.zeros([5]))

    # vmap both args
    result = vmapped(jnp.zeros((5, 3, 3)), jnp.zeros(5), 3)
    npt.assert_array_equal(result, np.full(5, 3))

    result = vmapped(jnp.zeros((2, 5, 3, 3)), jnp.zeros(5), 3)
    npt.assert_array_equal(result, np.full((2, 5), 3))

    result = vmapped(jnp.zeros((3, 3, 3)), jnp.zeros((4, 3)), 7)
    npt.assert_array_equal(result, np.full((4, 3), 7))

def test_varargs():
    """Test autovmap with variable arguments."""
    @autovmap(x=1, b=0)
    def foo(x, *args, b=0, c=1):
        return jnp.sum(x) + sum(args) + b + c

    result = foo(jnp.ones((3, 3)), 1, 2, 3, b=jnp.zeros((3,)))
    npt.assert_array_equal(result, np.array([10] * 3))

def test_pytree_uniform_rank():
    """Test autovmap with pytree inputs having uniform ranks."""
    def sum_ranks(objects, offset):
        total = offset
        for key in objects:
            total += objects[key].ndim
        return total

    names = 'abcde'
    ranks = [2, 5, 1, 2, 3]
    obj = {n: np.ones((2,) * r) for r, n in zip(ranks, names)}

    # Test with rank 1 expectation - reduces each array by 1 dimension
    total = autovmap(1, 0)(sum_ranks)(obj, 0)
    assert total.ndim == max(ranks) - 1
    npt.assert_array_equal(total, len(ranks))

    total = autovmap(1, 0)(sum_ranks)(obj, 1)
    npt.assert_array_equal(total, len(ranks) + 1)


def test_pytree_mixed_ranks():
    """Test autovmap with pytree inputs having mixed ranks."""
    def sum_ranks(objects, offset):
        total = offset
        for key in objects:
            total += objects[key].ndim
        return total

    names = 'abcd'
    ranks = [2, 5, 3, 2]
    obj = {n: np.ones((2,) * r) for r, n in zip(ranks, names)}

    # Test with rank 2 expectation
    total = autovmap(2, 0)(sum_ranks)(obj, 0)
    assert total.ndim == max(ranks) - 2
    npt.assert_array_equal(total, 2 * len(ranks))


def test_pytree_no_vmap():
    """Test that unmodified function works correctly."""
    def sum_ranks(objects, offset):
        total = offset
        for key in objects:
            total += objects[key].ndim
        return total

    names = 'abcd'
    ranks = [2, 5, 3, 2]
    obj = {n: np.ones((2,) * r) for r, n in zip(ranks, names)}
    total = sum_ranks(obj, 0)
    assert total == sum(ranks)


def test_pytree_rank_specification():
    """Test autovmap with pytree rank specification."""
    def sum_ranks(objects, offset):
        total = offset
        for key in objects:
            total += objects[key].ndim
        return total

    names = 'abcd'
    ranks = [2, 5, 3, 2]
    obj = {n: np.ones((2,) * r) for r, n in zip(ranks, names)}
    arg_ranks = {'a': 1, 'b': 2, 'c': 1, 'd': 0}

    total = autovmap(arg_ranks, 0)(sum_ranks)(obj, 0)
    npt.assert_array_equal(total, 1 + 2 + 1 + 0)

    total_rank = max(r - arg_ranks[n] for r, n in zip(ranks, names))
    assert total.ndim == total_rank


# Property-based tests using hypothesis
@given(
    base_shape=st.tuples(
        st.integers(min_value=1, max_value=5),  # matrix dim 1
        st.integers(min_value=1, max_value=5),  # matrix dim 2
    ),
    batch_shape=st.lists(
        st.integers(min_value=1, max_value=4),
        min_size=0, max_size=3
    ),
    scalar_val=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
@settings(deadline=None)  # Disable deadline due to JAX JIT compilation timing
def test_matrix_scalar_autovmap_property(base_shape, batch_shape, scalar_val):
    """Property test: autovmap should behave like manual vmap for matrix-scalar ops."""
    @autovmap(m=2, s=0)
    def matrix_trace_plus_scalar(m, s):
        return jnp.trace(m) + s

    # Create test arrays
    full_shape = tuple(batch_shape) + base_shape
    matrix = jnp.ones(full_shape)
    scalar = jnp.full(batch_shape, scalar_val) if batch_shape else scalar_val

    # Get autovmap result
    result = matrix_trace_plus_scalar(matrix, scalar)

    # Manually construct expected result using vmap
    def base_fn(m, s):
        return jnp.trace(m) + s

    if not batch_shape:
        # No batching needed
        expected = base_fn(matrix, scalar)
    else:
        # Apply vmap for each batch dimension
        vmapped_fn = base_fn
        for i in range(len(batch_shape)):
            in_axes = (0, 0 if jnp.ndim(scalar) > i else None)
            vmapped_fn = jax.vmap(vmapped_fn, in_axes=in_axes)
        expected = vmapped_fn(matrix, scalar)

    npt.assert_array_almost_equal(result, expected)


@given(
    rank=st.integers(min_value=0, max_value=3),
    batch_size=st.integers(min_value=1, max_value=3),
    dim_size=st.integers(min_value=2, max_value=4)
)
@settings(deadline=None)  # Disable deadline due to JAX JIT compilation timing
def test_rank_consistency_property(rank, batch_size, dim_size):
    """Property test: autovmap should consistently handle different ranks.

    Note: Uses deadline=None to avoid JAX JIT compilation timing issues.
    """
    @autovmap(x=rank)
    def compute_rank(x):
        return jnp.ndim(x)

    # Create array with specified rank and one batch dimension
    base_shape = (dim_size,) * rank if rank > 0 else ()  # scalars have empty shape
    full_shape = (batch_size,) + base_shape
    x = jnp.ones(full_shape)

    result = compute_rank(x)

    # Expected: rank should be preserved for each batch element
    assert jnp.all(result == rank)
    assert result.shape == (batch_size,)


@given(
    val=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
)
@settings(deadline=None)  # Disable deadline due to JAX JIT compilation timing
def test_broadcasting_property(val):
    """Property test: broadcasting with size-1 dimensions should work."""
    @autovmap(x=1, y=1)
    def add_vectors(x, y):
        return x + y

    # Test broadcasting: one vector has size 1 in batch dimension
    x = jnp.ones((1, 3))  # broadcasts
    y = jnp.full((4, 3), val)  # main shape

    result = add_vectors(x, y)
    expected = jnp.full((4, 3), 1.0 + val)

    npt.assert_array_almost_equal(result, expected)


def test_error_handling():
    """Test that appropriate errors are raised for invalid inputs."""
    @autovmap(x=2)
    def needs_matrix(x):
        return jnp.trace(x)

    # Should raise error when array rank is too small
    with pytest.raises(ValueError, match=r'Array "x" has rank 1 but expected at least 2'):
        needs_matrix(jnp.array([1, 2, 3]))  # rank 1, but needs rank 2

    # Should work fine with correct or higher rank
    result = needs_matrix(jnp.eye(3))  # rank 2 - no vmap
    assert result.shape == ()

    result = needs_matrix(jnp.ones((2, 3, 3)))  # rank 3 - vmap once
    assert result.shape == (2,)


def test_invalid_argument_names():
    """Test errors when specifying ranks for non-existent arguments."""
    def simple_func(x, y):
        return x + y

    # Test with keyword specification - non-existent argument name
    with pytest.raises(RuntimeError, match=r'Function has no argument "z"'):
        autovmap(z=1)(simple_func)  # z doesn't exist in simple_func


def test_no_ranks_specified():
    """Test autovmap with no ranks specified returns original function."""
    def original_func(x, y):
        return x * y + 1

    # No ranks specified - should return original function unchanged
    no_op_decorator = autovmap()
    decorated_func = no_op_decorator(original_func)

    # Should be able to call normally without any vmapping
    result = decorated_func(3.0, 2.0)
    assert result == 7.0


def test_pytree_with_mixed_ranks():
    """Test autovmap with pytrees containing mixed rank specifications."""
    @autovmap(data={'a': 1, 'b': 0})  # Mixed ranks in pytree
    def mixed_rank_func(data):
        return data['a'].sum() + data['b']

    # Create data where 'b' is scalar but 'a' needs vmapping
    test_data = {
        'a': jnp.ones((3, 5)),  # batch=3, base_rank=1 -> needs vmap
        'b': 2.0  # scalar, rank=0 -> should not be vmapped
    }

    result = mixed_rank_func(test_data)
    expected = jnp.array([7.0, 7.0, 7.0])  # 5*1 + 2 = 7 for each batch
    npt.assert_array_equal(result, expected)


def test_positional_and_keyword_ranks_conflict():
    """Test error when mixing positional and keyword rank specifications."""
    def dummy_func(x, y):
        return x + y

    # Should raise error when mixing positional and keyword args
    with pytest.raises(ValueError, match="Cannot mix positional and keyword rank specifications"):
        autovmap(1, x=2)(dummy_func)  # Both positional and keyword - not allowed
