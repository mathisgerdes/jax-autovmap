import jax.numpy as jnp
import numpy as np
import chex
from autovmap import autovmap
from unittest import TestCase


class VmapTests(TestCase):
    def assertAllEqual(self, x, y):
        x, y = map(np.asarray, (x, y))
        self.assertTrue(np.alltrue(x == y))

    def assertAllClose(self, x, y):
        x, y = map(np.asarray, (x, y))
        self.assertTrue(np.allclose(x, y))

    def assertAllTrue(self, t):
        self.assertTrue(np.alltrue(t))

    def test_dynamic_vmap(self):

        def foo(x, y, val=1):
            chex.assert_rank(x, 2)
            chex.assert_rank(y, 0)
            return val

        # args, kwargs
        possible_args = [
            ((2, 0), {}),
            ((2, 0, None), {}),
            (((2, 0),), {}),
            (((2, 0, None),), {}),
            (({'x': 2, 'y': 0},), {}),
            (({'x': 2, 'y': 0, 'val': None},), {}),
            (({'x': 2, 1: 0},), {}),  # not nice style but works...
            (({0: 2, 1: 0},), {}),
            ((), {'x': 2, 'y': 0}),
            ((), {'x': 2, 'y': 0, 'val': None}),
        ]

        for args, kwargs in possible_args:
            foo_dvm = autovmap(*args, **kwargs)(foo)

            # no vmap
            self.assertEqual(foo_dvm(jnp.zeros((3, 3)), 0.0), 1)
            # vmap second arg
            self.assertAllEqual(foo_dvm(jnp.zeros((3, 3)), [0.0, 1.0, 2.0]),
                                [1, 1, 1])
            # vmap first arg
            self.assertAllEqual(foo_dvm(jnp.zeros((5, 3, 3)), 0),
                                [1, 1, 1, 1, 1])
            self.assertAllEqual(foo_dvm(jnp.zeros((5, 3, 3)), 0, val=0),
                                [0, 0, 0, 0, 0])
            self.assertAllEqual(foo_dvm(jnp.zeros((5, 3, 3)), 0, 0),
                                [0, 0, 0, 0, 0])
            # vmap both args
            self.assertAllEqual(
                foo_dvm(jnp.zeros((5, 3, 3)), jnp.zeros(5), 3),
                np.full(5, 3))
            self.assertAllEqual(
                foo_dvm(jnp.zeros((2, 5, 3, 3)), jnp.zeros(5), 3),
                np.full((2, 5), 3))
            self.assertAllEqual(
                foo_dvm(jnp.zeros((3, 3, 3)), jnp.zeros((4, 3)), 7),
                np.full((4, 3), 7))
