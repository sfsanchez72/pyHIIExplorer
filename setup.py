#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

def readme():
   with open("README.rst") as f:
       return f.read()

DATA_DIRNAME = "data"
SCRIPTS_DIRNAME = "bin"
NOTEBOOKS_DIRNAME = "notebooks"

all_packages = find_packages()
packages_data = {package: [f"{DATA_DIRNAME}/*"]+[f"{os.path.join(DATA_DIRNAME, sub)}/*" for root, subs, files in os.walk(os.path.join(package, DATA_DIRNAME)) for sub in subs] for package in all_packages if os.path.isdir(os.path.join(package, DATA_DIRNAME))}
scripts = [os.path.join(SCRIPTS_DIRNAME, script_name) for script_name in os.listdir(SCRIPTS_DIRNAME) if script_name.endswith(".py")]

setup(
    name='pyHIIExplorer',
    version='0.1.0',    
    description='A new version of HIIExplorer for python',
    url='https://github.com/sfsanchez72/pyHIIExplorer',
    author='Sebastian F. Sanchez & Alejandra Z. Lugo',
    author_email='sfsanchez@astro.unam.mx',
    license='BSD 2-clause',
    packages=['pyHIIExplorer'],
    install_requires=['numpy',
                      'scipy',
                      'astropy',
                      'matplotlib',
                      'scikit-image',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    include_package_data=True,
    package_data=packages_data,
    scripts=scripts,
    zip_safe=False,
    test_suite="nose.collector",
    tests_require=["nose"], 
)
