[![Tests](https://github.com/mathisgerdes/autovmap/actions/workflows/python-pytest.yml/badge.svg)](https://github.com/mathisgerdes/autovmap/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/jax-autovmap)](https://pypi.org/project/jax-autovmap/)
# Automatically broadcast via JAX vmap

Automatically broadcast a function that takes inputs of specific ranks to accept any batch dimensions using `jax.vmap`.
See the documentation of [JAX](https://github.com/google/jax) for more information about `vmap`.

This module defines a decorator function which takes as input the base ranks of the arguments of a function.
The transformed function takes any broadcastable combination of batched inputs and automatically applies `jax.vmap` as required.
One could equivalently apply `vmap` by hand, however the in-axes have to be chosen differently for different inputs.
The decorator takes care of this automatically; if the underlying ranks are known, batch dimensions can be inferred.

## Examples

Consider the following function which takes numeric arguments with fixed and known ranks as input:
```python
import jax.numpy as jnp

def foo(s, v, m):  # s - scalar, v - vector, m - matrix
    return v @ m @ v + s * v.size
```
If we have inputs of appropriate rank, the function can be applied without problem:
```python
s = jnp.array(2.0)    # scalar
v = jnp.ones(3)       # vector
m = jnp.ones((3, 3))  # matrix
foo(s, v, m)   # returns 15.0
```
Assume now, however, that we have 5 matrices and 5 vectors for which we want to apply the above function as a batch.
Many numpy functions can take inputs with leading batch dimensions, but here we have an issue because `v @ m @ v` requires `m` to be a matching matrix.
```python
s = jnp.array(2.0)
v = jnp.ones((5, 3))
m = jnp.ones((5, 3, 3))
foo(s, v, m)  # throws TypeError
```
There are multiple possible ways we can solve this

- We could try to write the function more carefully, so it can take both single and batch inputs.
  Or we could always require the first dimension to be a batch index.
- Given known inputs, we can transform our function: `jax.jit(foo, (None, 0, 0))`.
  However, the axes change depending on the inputs.
  If we want to expose functions that accept batched inputs to the user, we need to have some clear naming scheme (to indicate which arguments are batched).

Sometimes, the best solution is one of the above.
This module provides another more flexible solution.
If the ranks the function *wants* are known, we can derive which arguments have (leading) batch dimensions.
Based on that, we can apply `jax.vmap` appropriately.
That is exactly what the `auto_vmap` wrapper does.
Thanks to `jax.jit`, after the transformed function is JIT-compiled, there is no price to pay for this extra flexibility since it only depends on the statically-known input shapes.

We can define the more flexible function as follows:
```python
from jax_autovmap import auto_vmap

@auto_vmap(s=0, v=1, m=2)
def foo(s, v, m):
    return v @ m @ v + s * v.size

foo(s, v, m)  # returns [15. 15. 15. 15. 15.]
```
The ranks can be specified by keyword argument as above, or positionally (in this case `@auto_vmap(0, 1, 2)`).
This does not have to be applied to all input arguments.
They can either be omitted or, if ranks are given positionally, specified as `None`.

If the arguments are pytrees (python structures of arrays) and the rank is a single integer, all constituents (leaves) are assumed to have that rank.
Alternatively, the rank can be a matching pytree, just like the `in_axes` in `jax.vmap`:
```python
@auto_vmap({'s': 0, 'v': 1, 'm': 2})
def foo(inputs):
    return inputs['v'] @ inputs['m'] @ inputs['v'] + inputs['s']

foo(dict(s=s, v=v, m=m))
```

Just like NumPy broadcasting, it is also allowed that one of the "vmap'ed" arguments has length 1:
```python
s = jnp.array(2.0)
v = jnp.ones((7, 1, 3))  # broadcast second axis with vmap over 5 values of `m`
m = jnp.ones((7, 5, 3, 3))

@auto_vmap(s=0, v=1, m=2)
def foo(s, v, m):
    return v @ m @ v + s

foo(s, v, m)  # shape (7, 5)
```
