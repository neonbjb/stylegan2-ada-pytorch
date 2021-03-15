from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='upfirdn2d_plugin',
    ext_modules=[
        CUDAExtension('upfirdn2d_plugin', [
            'upfirdn2d.cpp',
            'upfirdn2d_cu.cu',
        ],
        extra_cuda_cflags=['--use_fast_math'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)