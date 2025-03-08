from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define Cython extension
extensions = [
    Extension("fluxion.math_util_fast", ["fluxion/math_util_fast.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name="fluxion",
    version="0.0.1",
    packages=["fluxion"],
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
)
