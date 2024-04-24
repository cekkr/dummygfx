# setup.py
# python3 setup.py build_ext

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("dummygfx.py")
)
