from __future__ import annotations

import inspect
from functools import partial, wraps
from inspect import signature
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

__version__ = '0.3.0'


def _is_numeric(value: Any) -> bool:
    """True if input is a numpy/jax array or scalar value."""
    return isinstance(
        value, (jax.Array, np.ndarray, np.generic, float, complex, bool, int)
    ) or hasattr(value, "__jax_array__")


def _validate_rank_args(ranks_pos: tuple, ranks_kw: dict[str, Any]) -> Union[tuple, dict[str, Any]]:
    """Validate and return the appropriate rank specification."""
    if len(ranks_pos) > 0:
        if len(ranks_kw) > 0:
            raise ValueError(
                'Cannot mix positional and keyword rank specifications. '
                f'Got positional: {ranks_pos}, keyword: {ranks_kw}')
        return ranks_pos
    elif len(ranks_kw) > 0:
        return ranks_kw
    else:
        # No ranks specified - return identity decorator
        return None


def _vmap_count(arg: Any, rank: int, name: str) -> int:
    """Get number of leading dimensions we need to vmap over."""
    ndim = jnp.ndim(arg)
    if ndim == rank:
        return 0  # no vmap, rank matches expectation
    if ndim < rank:
        raise ValueError(
            f'Array "{name}" has rank {ndim} but expected at least {rank}. '
            f'Got shape {jnp.shape(arg)}.')
    return ndim - rank


def _vmap_axes(vmap_rank: int, level: int) -> Optional[int]:
    """Choose vmap axis (None or 0) given vmap rank and level."""
    return 0 if vmap_rank > level else None


def _broadcast_vmap(fn: Callable, in_axes: Sequence[Any]) -> Callable:
    """Functions like vmap, except axes are allowed to be size 1 for broadcasting.

    Assumes in_axes are all either 0 or None.
    """
    def _wrapped(*args):
        args_squeezed = []
        axes_squeezed = []
        do_squeeze = False
        for arg, ax in zip(args, in_axes):
            # nothing to do for arguments that are not auto-vmap'ed
            if ax is None:
                args_squeezed.append(arg)
                axes_squeezed.append(ax)
                continue

            arg_leaves, treedef = jax.tree.flatten(arg)
            if isinstance(ax, int):
                # apply int to all leaves
                ax_leaves = [ax] * len(arg_leaves)
            else:
                # must be matching pytree of ints
                ax_leaves = treedef.flatten_up_to(ax)

            for i, (arg_leaf, ax_leaf) in enumerate(zip(arg_leaves, ax_leaves, strict=True)):
                if jnp.shape(arg_leaf) == ():
                    assert ax_leaf is None, 'cannot vmap over scalar value'
                elif ax_leaf is not None:
                    if len(arg_leaf) == 1:
                        arg_leaves[i] = jnp.squeeze(arg_leaf, 0)
                        ax_leaves[i] = None  # exclude from vmap
                    # only use squeezed args if at least one array has len > 1
                    else:
                        do_squeeze = True

            args_squeezed.append(treedef.unflatten(arg_leaves))
            axes_squeezed.append(treedef.unflatten(ax_leaves))
        if do_squeeze:
            return jax.vmap(fn, in_axes=axes_squeezed)(*args_squeezed)
        return jax.vmap(fn, in_axes=in_axes)(*args)
    return _wrapped


def _collect_ranks(
        fun: Callable,
        rspec: Union[dict[str, Optional[int]], Sequence[Optional[int]]]
    ) -> tuple[inspect.Signature, dict[str, int]]:
    """Parse user provided ranks specification given function."""
    sig = signature(fun)
    arg_list = list(sig.parameters.keys())
    ranks: dict[str, int] = {}

    if isinstance(rspec, dict):
        for name, rank in rspec.items():
            if name not in arg_list:
                raise RuntimeError(
                    f'Function has no argument "{name}". '
                    f'Available arguments: {", ".join(arg_list)}')
            if rank is not None:
                ranks[name] = rank
    else:
        for i, rank in enumerate(rspec):
            if rank is not None and i < len(arg_list):
                ranks[arg_list[i]] = rank
            elif rank is not None:
                raise RuntimeError(
                    f'Too many positional ranks specified. '
                    f'Function has {len(arg_list)} arguments but got rank for position {i}')

    return sig, ranks


def _vmap_wrapped(fun: Callable, sig: inspect.Signature, vmap_count: Sequence[Any], *args: Any) -> Any:
    """Wrap function in vmap according to counts needed per argument."""
    vmap_depth = max(jax.tree.leaves(vmap_count))

    def wrapped(*all_args):
        # Effectively converts function which may have keyword-only arguments
        # into a purely positional function.
        positional = []
        kwargs = dict()
        for arg, param in zip(all_args, sig.parameters.values()):
            if param.kind == param.KEYWORD_ONLY:
                kwargs[param.name] = arg
            elif param.kind == param.VAR_POSITIONAL:
                positional.extend(arg)
            else:
                positional.append(arg)
        bound = sig.bind(*positional, **kwargs)
        return fun(*bound.args, **bound.kwargs)

    for level in range(vmap_depth):
        axes = jax.tree.map(partial(_vmap_axes, level=level), vmap_count)
        wrapped = _broadcast_vmap(wrapped, in_axes=axes)

    return wrapped(*args)


def autovmap(*ranks_pos: Union[int, Any, None],
             **ranks_kw: Union[int, Any, None]) -> Callable[[Callable], Callable]:
    """Automatically vmap over function arguments with specified ranks.

    Transforms a function to accept batched inputs by applying jax.vmap
    automatically based on the expected rank of each argument.

    Args:
        *ranks_pos: Positional specification of argument ranks (int, pytree, or None)
        **ranks_kw: Keyword specification of argument ranks

    Returns:
        Decorator that transforms functions to accept batched inputs

    Examples:
        @autovmap(2, 0)  # matrix, scalar
        def det_scale(mat, s):
            return jnp.linalg.det(mat) * s

        @autovmap(mat=2, s=0)  # equivalent keyword form
        def det_scale(mat, s):
            return jnp.linalg.det(mat) * s
    """
    arg_rank = _validate_rank_args(ranks_pos, ranks_kw)
    if arg_rank is None:
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

                if _is_numeric(arg):
                    vmap_count = _vmap_count(arg, rank, name)
                elif isinstance(rank, int):
                    leaves, treedef = jax.tree.flatten(arg)
                    vmap_count = tuple(
                        _vmap_count(leave, rank, name) for leave in leaves)
                    vmap_count = treedef.unflatten(vmap_count)
                else:
                    leaves, treedef = jax.tree.flatten(arg)
                    rank_leaves = jax.tree.leaves(rank)
                    vmap_count = tuple(
                        _vmap_count(leave, r, name)
                        for leave, r in zip(leaves, rank_leaves))
                    vmap_count = treedef.unflatten(vmap_count)

                vmap_counts.append(vmap_count)
                in_args.append(arg)

            return _vmap_wrapped(fun, sig, vmap_counts, *in_args)

        return fun_dynamic

    return wrapper


# for backwards compatibility
auto_vmap = autovmap


__all__ = ['auto_vmap', 'autovmap']
