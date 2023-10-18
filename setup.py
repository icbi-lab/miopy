#!/usr/bin/python3
from setuptools import setup, find_packages

# Define your package name and other metadata
package_name = 'miopy'
version = '1.3.6'
author = 'Pablo Monfort Lanzas'
author_email = 'pablo.monfort@i-med.ac.at'
description = 'MIOPY: Python tool to study miRNA-mRNA relationships.'
github_url = 'https://github.com/icbi-lab/miopy'

# Read the contents of README.md for the long description
with open('README.rst', 'r') as f:
    long_description = f.read()

# Read the list of requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f]

setup(
    name=package_name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=github_url,
    packages=find_packages(),
    scripts=['bin/mio_correlation.py'],
    license='LICENSE.txt',  # Make sure you have the license file
    install_requires=requirements,
    include_package_data=True,
    package_data={'miopy': ['data/*', 'Rscript/*.r']},
)
