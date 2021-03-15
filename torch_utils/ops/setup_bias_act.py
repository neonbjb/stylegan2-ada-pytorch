from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bias_act_plugin',
    ext_modules=[
        CUDAExtension('bias_act_plugin', [
            'bias_act.cpp',
            'bias_act_cu.cu',
        ],
        extra_cuda_cflags=['--use_fast_math'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)