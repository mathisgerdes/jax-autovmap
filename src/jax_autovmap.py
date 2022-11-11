from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from inspect import signature
from functools import wraps, partial
from typing import Optional, Union, Callable, Any

__version__ = '0.1.0'

_CONVERT_ARRAY_TYPES = (list, int, float, complex)


def _vmap_count(arg, rank, name):
    """Get number of leading dimensions we need to vmap over"""
    if arg.ndim == rank:
        return 0  # no vmap, rank matches expectation
    if arg.ndim < rank:
        raise ValueError(
            f'Rank of array passed to `{name}` too small. '
            f'Got shape {arg.shape} but expected '
            f'at least rank {rank}.')
    return arg.ndim - rank


def _vmap_axes(vmap_rank, level):
    """Choose vmap axis (None or 0) given vmap rank and level."""
    return 0 if vmap_rank > level else None


def _collect_ranks(
        fun: Callable,
        rspec: Union[dict[str, Optional[int]], tuple[Optional[int], ...]]) \
        -> tuple[inspect.Signature, dict]:
    """Parse user provided ranks specification given function."""
    sig = signature(fun)
    arg_list = list(sig.parameters.keys())
    if isinstance(rspec, dict):
        ranks = {}
        for name, rank in rspec.items():
            if name not in arg_list:
                raise RuntimeError(
                    f'function {fun} has no argument called "{name}" '
                    f'to dynamically vmap over.')
            if rank is not None:
                ranks[name] = rank
    else:
        ranks = {arg_list[i]: rank
                 for i, rank in enumerate(rspec)
                 if rank is not None}
    for name in ranks:
        if name not in arg_list:
            fun_name = fun.__name__ if hasattr(fun, '__name__') else fun
            raise RuntimeError(
                f'Specified argument "{name}" does not appear as a '
                f'parameter of the function "{fun_name}"')

    return sig, ranks


def _vmap_wrapped(fun, sig, vmap_count, *args):
    """Wrap function in vmap according to counts needed per argument."""
    vmap_depth = max(tree_util.tree_leaves(vmap_count))

    def wrapped(*all_args):
        # Effectively converts function which may have keyword-only arguments
        # into a purely positional function.
        positional = []
        kwargs = dict()
        for arg, param in zip(all_args, sig.parameters.values()):
            if param.kind == param.KEYWORD_ONLY:
                kwargs[param.name] = arg
            else:
                positional.append(arg)
        bound = sig.bind(*positional, **kwargs)
        return fun(*bound.args, **bound.kwargs)

    for level in range(vmap_depth):
        axes = tree_util.tree_map(partial(_vmap_axes, level=level), vmap_count)
        wrapped = jax.vmap(wrapped, in_axes=axes)

    return wrapped(*args)


def auto_vmap(*ranks_pos: Union[int, Any, None],
              **ranks_kw: Union[int, Any, None]):
    """Automatically vmap over arguments of a function.

    Given a function with some arguments that have known
    fundamental ranks, we want make it take any batched inputs.
    This is meant to correspond to the broadcasting behaviour of numpy.

    For example, the fundamental rank for numpy.sin is 0 as it is defined
    for scalar values. If the input array has shape S then so does the output,
    the function call is automatically boradcast over the input dimensions.

    This function is intended as a decorator, which transforms the function
    to automatically broadcast. For example, if ``foo`` is a function
    which takes as input a matrix (of rank 2) and a scalar (of rank 0),
    it can be written as follows:

    .. code-block:: python

            @dynamic_vmap(2, 0)
            def _foo(mat, s):
                return jnp.linalg.det(mat) * s

    Instead of positionally giving the fundamental ranks, they can also
    be specified by name:

    .. code-block:: python

            @dynamic_vmap(mat=2, s=0)
            def _foo(mat, s):
                return jnp.linalg.det(mat) * s

    If an argument should not be vmapped over, the rank can be set to
    ``None`` or omitted.

    The ranks should typically be positive integers.
    Just like vmap, they can also be pytrees of integers that mache the
    corresponding input arguments.

    Examples:
        First, define an example function where x must be
        a matrix (rank 2) and y a scalar (rank 0):

        >>> def foo(mat, s):
        ...   return jnp.linalg.det(mat) * s
        >>> foo = auto_vmap(x=2, y=0)(foo)
        >>> mat, scalars = jnp.zeros((3, 3)), [0.0, 1.0, 2.0] # some inputs
        >>> out = foo(mat, scalars) # will automatically vmap over second arg
        >>> len(out) == len(scalars) # one output per scalar in inputs
        True
    """
    if len(ranks_pos) > 0:
        assert len(ranks_kw) == 0, \
            f'ranks must either be given positionally or via ' \
            f'keyword arguments, but got both positional arguments ' \
            f'{ranks_pos} and keyword arguments {ranks_kw}'
        arg_rank = ranks_pos
    elif len(ranks_kw) > 0:
        arg_rank = ranks_kw
    else:
        return lambda fun: fun

    def wrapper(fun):
        sig, ranks = _collect_ranks(fun, arg_rank)

        @wraps(fun)
        def fun_dynamic(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # get number of times arguments need to be mapped
            vmap_counts = []
            in_args = []  # possibly convert to jnp.ndarray
            for name, arg in bound.arguments.items():
                if arg is None or name not in ranks:
                    # do nothing with this argument
                    vmap_counts.append(0)
                    in_args.append(arg)
                    continue

                rank = ranks[name]
                if isinstance(arg, _CONVERT_ARRAY_TYPES):
                    arg = jnp.asarray(arg)

                if isinstance(arg, (jnp.ndarray, np.ndarray)):
                    vmap_count = _vmap_count(arg, rank, name)
                elif isinstance(rank, int):
                    leaves, treedef = tree_util.tree_flatten(arg)
                    vmap_count = tuple(
                        _vmap_count(leave, rank, name) for leave in leaves)
                    vmap_count = treedef.unflatten(vmap_count)
                else:
                    leaves, treedef = tree_util.tree_flatten(arg)
                    rank_leaves = tree_util.tree_leaves(rank)
                    vmap_count = tuple(
                        _vmap_count(leave, r, name)
                        for leave, r in zip(leaves, rank_leaves))
                    vmap_count = treedef.unflatten(vmap_count)

                vmap_counts.append(vmap_count)
                in_args.append(arg)

            return _vmap_wrapped(fun, sig, vmap_counts, *in_args)

        return fun_dynamic

    return wrapper


__all__ = ['auto_vmap']
