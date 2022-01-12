#!/usr/bin/python3
from setuptools import setup

setup(
   name='miopy',
   version='1.0.01',
   author='Pablo Monfort Lanzas',
   author_email='pablo.monfort@i-med.ac.at',
   packages=['miopy', ],
   scripts=['bin/mio_correlation.py',],
   url='https://gitlab.i-med.ac.at/cbio/miopy',
   license='LICENSE.txt',
   description='',
   long_description=open('README.rst').read(),
   long_description_content_type='text/markdown',
   install_requires=[
       "pandas",
       "scipy",
       "numpy",
       "ranky",
       "pandarallel",
       "statsmodels",
       "scikit-learn==0.24.0",
       "lifelines",
       "argparse",
       "eli5",
       "Cython",
       "scikit-survival",

   ],
   include_package_data=True,
package_data={'': ['data/*', "Rscript/*.r", "dataset/*.csv"],
            },
)
