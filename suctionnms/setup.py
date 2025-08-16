# // Code Written by Hanwen Cao

from distutils.core import setup
import setuptools
from Cython.Build import cythonize
import numpy

setup(
    version='1.0.0',
    description='suction pose non-maximum supression',
    author='Hanwen Cao',
    author_email='hwcao17@gmail.com',
    url='https://github.com/intrepidChw/suctionnms.git',
    name = 'suction_nms',
    ext_modules=cythonize("suction_nms.pyx"),
    include_dirs=[numpy.get_include()]
)
