from setuptools import find_packages, setup
from Cython.Build import cythonize
import torch
import numpy
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


setup(
    name="CAP",
    ext_modules=[
        CppExtension(
            name='cpu_nms',
            sources=['cpu_nms.cpp'],
            extra_compile_args={'cxx': []}
        ),
        # CUDAExtension(
        #     name='gpu_nms',
        #     sources=['gpu_nms.cpp', 'nms_kernel.cu'],
        #     define_macros=[('WITH_CUDA', None)],
        #     extra_compile_args={'cxx': [],'nvcc': [
        #         '-D__CUDA_NO_HALF_OPERATORS__',
        #         '-D__CUDA_NO_HALF_CONVERSIONS__',
        #         '-D__CUDA_NO_HALF2_OPERATORS__',
        #     ]},
        #     include_dirs=['gpu_nms.hpp']
        # )
    ],
    cmdclass={'build_ext': BuildExtension},
    include_dirs=[numpy.get_include()]
)
