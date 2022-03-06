from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='toad',
    version='0.1',
    description='Tipping and other abrupt events detector',
    long_description=readme(),
    author='Sina Loriani',
    author_email='sina.loriani@pik-potsdam.de',
    packages=find_packages(),
    install_requires = [
        'numpy',
        'matplotlib',
        'scipy',
        'sklearn',
        'xarray',
        'scipy',
        'seaborn',
        'netcdf4',
        'ipykernel',
        'cartopy',
        'dask',
        'distributed'
    ]
)