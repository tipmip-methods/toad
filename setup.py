from setuptools import setup, find_packages
from _version import __version__

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
    packages=find_packages(),
    install_requires = [
        'numpy',
        'matplotlib',
        'scipy',
        #'sklearn',
        'scikit-learn',
        'xarray',
        'scipy',
        'seaborn',
        'netcdf4',
        'ipykernel',
        'dask',
        'distributed'
    ]
)