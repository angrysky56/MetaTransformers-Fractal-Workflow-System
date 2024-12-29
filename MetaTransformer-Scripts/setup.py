from setuptools import setup, find_packages

setup(
    name="metatransformer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'neo4j',
    ],
)