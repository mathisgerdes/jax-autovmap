from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from inspect import signature, getfullargspec
from functools import wraps
from typing import Union

_CONVERT_ARRAY_TYPES = (list, int, float, complex, chex.Array)

_arg_type = Union[dict[Union[str, int], int], int, tuple[int, ...], None]
_kwarg_type = int


def autovmap(*args: _arg_type, **kwargs: _kwarg_type):
    """Decorator: dynamically vmap over arguments of a function.

    Possible signatures:
        First, define a dummy function as an example where ``x`` is to be of
        rank 2 (a matrix) and ``y`` is a scalar (rank 0):
        >>> def _foo(x, y, val=1):
        ...   chex.assert_rank(x, 2)
        ...   chex.assert_rank(y, 0)
        ...   return val

        All following lines are equivalent (and could equivalently be used
        as decorators in the function definition above). The various ways
        should not be mixed, however.
        >>> foo1 = autovmap(2, 0)(_foo)
        >>> foo2 = autovmap(2, 0, None)(_foo)
        >>> foo3 = autovmap((2, 0))(_foo)
        >>> foo4 = autovmap(x=2, y=0)(_foo)
        >>> foo5 = autovmap({0: 2, 1: 0})(_foo)
        >>> foo6 = autovmap({'x': 2, 'y': 0})(_foo)
        >>> foo6(jnp.zeros((3, 3)), 0.0)  # no vmap
        1
        >>> out = foo6(jnp.zeros((3, 3)), [0.0, 1.0, 2.0])  # vmap second arg
        >>> print(out)
        [1 1 1]
    """
    if len(args) > 0:
        assert len(kwargs) == 0, \
            f'Ranks must either be given by a tuple, a dictionary, ' \
            f'positionally or via keyword arguments. Cannot mix any two ' \
            f'variants. Called function with positional arguments:' \
            f'\n{args}\n...and keyword arguments:\n{kwargs}'
        if len(args) == 1 and not isinstance(args[0], int):
            arg_rank, = args
        else:
            arg_rank = args
    elif len(kwargs) > 0:
        arg_rank = kwargs
    else:
        return lambda fun: fun

    def wrapper(fun):
        sig = signature(fun)
        arg_list = getfullargspec(fun).args
        if isinstance(arg_rank, dict):
            # `name` can be an argument name (str) or an argument index
            ranks = {name if isinstance(name, str) else arg_list[name]: rank
                     for name, rank in arg_rank.items()
                     if rank is not None}
        else:
            ranks = {arg_list[i]: rank
                     for i, rank in enumerate(arg_rank)
                     if rank is not None}
        for name in ranks:
            if name not in arg_list:
                fun_name = fun.__name__ if hasattr(fun, '__name__') else fun
                raise RuntimeError(
                    f'Specified argument "{name}" does not appear as a '
                    f'parameter of the function "{fun_name}"')

        @wraps(fun)
        def fun_dynamic(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if len(bound.kwargs) != 0:
                raise RuntimeError(
                    'Dynamic vmap does not support pure keyword arguments')

            vmap_ranks = []
            in_args = []  # possibly convert to jnp.ndarray
            for arg, name in zip(bound.args, arg_list):
                if arg is None:
                    vmap_ranks.append(0)
                elif name in ranks:
                    rank = ranks[name]
                    if isinstance(arg, _CONVERT_ARRAY_TYPES):
                        arg = jnp.asarray(arg)
                    elif not isinstance(arg, chex.Array):
                        raise ValueError(
                            f'Cannot vmap over object of type "{type(arg)}" '
                            f'passed to parameter "{name}"')
                    if arg.ndim == rank:
                        vmap_ranks.append(0)
                    else:
                        # need to perform vmap over this arg
                        if arg.ndim < rank:
                            raise ValueError(
                                f'Rank of array passed to `{name}` too small. '
                                f'Got shape {arg.shape} but expected '
                                f'at least rank {rank}.')
                        vmap_ranks.append(arg.ndim - rank)
                else:
                    vmap_ranks.append(0)
                in_args.append(arg)
            vmap_depth = max(vmap_ranks)
            v_fun = fun
            for d in range(vmap_depth):
                v_fun = jax.vmap(
                    v_fun,
                    in_axes=tuple(0 if r > d else None for r in vmap_ranks))
            return v_fun(*in_args)

        return fun_dynamic

    return wrapper
