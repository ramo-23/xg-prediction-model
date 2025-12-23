from setuptools import setup, find_packages

setup(
    name='xg_prediction_model',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
