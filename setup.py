from setuptools import setup


setup(
    name='autovmap',
    version='1.0',
    description='Automatically apply vmap given base dimensions of input.',
    author='Mathis Gerdes',
    author_email='MathisGerdes@gmail.com',
    packages=['.'],
    python_requires='>=3.7',
    install_requires=[
        'jax>=0.2.20',
        'jaxlib>=0.1.69',
        'numpy',
        'chex',
    ],
)
