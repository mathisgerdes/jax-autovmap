import jax.numpy as jnp
import numpy as np
from jax_autovmap import auto_vmap
from unittest import TestCase


class AutoVmapTests(TestCase):
    def assertAllEqual(self, x, y):
        x, y = map(np.asarray, (x, y))
        self.assertTrue(np.alltrue(x == y))

    def assertAllClose(self, x, y):
        x, y = map(np.asarray, (x, y))
        self.assertTrue(np.allclose(x, y))

    def assertAllTrue(self, t):
        self.assertTrue(np.alltrue(t))

    def test_basic(self):

        def foo(x, y, val=1):
            self.assertEqual(x.ndim, 2)
            self.assertEqual(y.ndim, 0)
            return val

        # args, kwargs
        possible_args = [
            ((2, 0), {}),
            ((2, 0, None), {}),
            ((), {'x': 2, 'y': 0}),
            ((), {'x': 2, 'y': 0, 'val': None}),
        ]

        for args, kwargs in possible_args:
            vmapped = auto_vmap(*args, **kwargs)(foo)

            # no vmap
            self.assertEqual(vmapped(jnp.zeros((3, 3)), 0.0), 1)
            # vmap second arg
            self.assertAllEqual(vmapped(jnp.zeros((3, 3)), [0.0, 1.0, 2.0]),
                                [1, 1, 1])
            # vmap first arg
            self.assertAllEqual(vmapped(jnp.zeros((5, 3, 3)), 0),
                                [1, 1, 1, 1, 1])
            self.assertAllEqual(vmapped(jnp.zeros((5, 3, 3)), 0, val=0),
                                [0, 0, 0, 0, 0])
            self.assertAllEqual(vmapped(jnp.zeros((5, 3, 3)), 0, 0),
                                [0, 0, 0, 0, 0])
            # vmap both args
            self.assertAllEqual(
                vmapped(jnp.zeros((5, 3, 3)), jnp.zeros(5), 3),
                np.full(5, 3))
            self.assertAllEqual(
                vmapped(jnp.zeros((2, 5, 3, 3)), jnp.zeros(5), 3),
                np.full((2, 5), 3))
            self.assertAllEqual(
                vmapped(jnp.zeros((3, 3, 3)), jnp.zeros((4, 3)), 7),
                np.full((4, 3), 7))

    def test_pytree(self):
        def sum_ranks(objects, offset):
            total = offset
            for key in objects:
                total += objects[key].ndim
            return total

        names = 'abcde'
        ranks = [2, 5, 1, 2, 3]
        obj = {n: np.ones((2,)*r) for r, n in zip(ranks, names)}

        # each is rank 5 so the sum is just the number of arguments
        total = auto_vmap(1, 0)(sum_ranks)(obj, 0)
        self.assertEqual(total.ndim, max(ranks) - 1)
        self.assertAllEqual(total, len(ranks))
        total = auto_vmap(1, 0)(sum_ranks)(obj, 1)
        self.assertAllEqual(total, len(ranks) + 1)

        # rank is 2 so two times the number of arguments
        names = 'abcd'
        ranks = [2, 5, 3, 2]
        obj = {n: np.ones((2,)*r) for r, n in zip(ranks, names)}
        total = auto_vmap(2, 0)(sum_ranks)(obj, 0)
        self.assertEqual(total.ndim, max(ranks) - 2)
        self.assertAllEqual(total, 2 * len(ranks))

        # check unmodified code does the right thing
        names = 'abcd'
        ranks = [2, 5, 3, 2]
        obj = {n: np.ones((2,) * r) for r, n in zip(ranks, names)}
        total = sum_ranks(obj, 0)
        self.assertEqual(total, sum(ranks))

        # test specification as pytree
        names = 'abcd'
        ranks = [2, 5, 3, 2]
        obj = {n: np.ones((2,) * r) for r, n in zip(ranks, names)}
        arg_ranks = {'a': 1, 'b': 2, 'c': 1, 'd': 0}
        total = auto_vmap(arg_ranks, 0)(sum_ranks)(obj, 0)
        self.assertAllEqual(total, 1 + 2 + 1 + 0)
        total_rank = max(r - arg_ranks[n] for r, n in zip(ranks, names))
        self.assertAllEqual(total.ndim, total_rank)
