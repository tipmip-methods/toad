from setuptools import setup, find_packages
from toad._version import __version__

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='toad',
    version=__version__,
    description='Tipping and other abrupt events detector',
    long_description=readme(),
    author='Sina Loriani',
    author_email='sina.loriani@pik-potsdam.de',
    maintainer='Jakob Harteg',
    maintainer_email='jakob.harteg@pik-potsdam.de',
    url='https://github.com/tipmip-methods/toad',
    packages=find_packages(exclude=["tutorials"]),
    # TODO: add version bounds for dependencies
    install_requires = [
        'numpy',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'xarray',
        'scipy',
        'seaborn',
        'netcdf4',
        'ipykernel',
        'dask',
        'distributed',
        'cartopy',
        'hdbscan'
    ]
)