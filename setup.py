from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = [r.rstrip('\n') for r in f.readlines()]

setup(
    name='DSToolbox',
    version='0.0.1',
    description='Datascience toolbox',
    url='https://www.asidatascience.com',
    author='Andris Piebalgs',
    author_email='andris.piebalgs@outlook.com',
    packages=find_packages(),
    install_requires=requirements
)