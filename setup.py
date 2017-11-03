from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = [r.rstrip('\n') for r in f.readlines()]

setup(
    name='ASI_IP_library',
    version='0.0.1',
    description='Python library for ASI and IP',
    url='https://www.asidatascience.com',
    author='ASI Data Science',
    author_email='andris.p@asidatascience.com',
    packages=find_packages(),
    install_requires=requirements
)