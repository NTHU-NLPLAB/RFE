from setuptools import setup, find_packages

setup(
    name='data_helper',
    version='0.0.1',
    packages=find_packages(
        include = ["teufel_patterns"],
        exclude=[]
        )
    )