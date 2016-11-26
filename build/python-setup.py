from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

setup(ext_modules = cythonize(Extension("fastnet",
           ["fastnet.pyx"],                 # our Cython source
           language="c++",             # generate C++ code
           extra_compile_args=['-std=c++11', '-mmacosx-version-min= 10.9'],
           extra_link_args=['-std=c++11', '-mmacosx-version-min= 10.9'],
           library_dirs=["/Users/rtorres/Tools/lib/fastnet/lib"],
           include_dirs=["/Users/rtorres/Tools/fastnet"],
           libraries=["neuralnet", "training"])
      ))
