#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md',encoding="utf8") as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Genomics and Machine Learning lab",
    author_email='duy.pham@uq.edu.au',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A downstream analysis toolkit for Spatial Transcriptomic data",
    entry_points={
        'console_scripts': [
            'stlearn=stlearn.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='stlearn',
    name='stlearn',
    packages=find_packages(include=['stlearn', 'stlearn.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/BiomedicalMachineLearning/stLearn',
    version='0.2.6',
    zip_safe=False,
)
