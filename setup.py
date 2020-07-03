#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
	name='pracopt',
	version='0.0.1dev',
	authors='Rob Sumner',
	author_email='rob_sumner1@hotmail.co.uk',
	packages=find_packages(),
	license='All Rights Reserved',
	description=('Practical optimisation tools.'),
	python_requires='>=3.7',
	install_requires=[],
	tests_require=['pytest>=4.0']
)