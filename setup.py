from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='test_cuda',
    ext_modules=[
        CUDAExtension('test_cuda', [
            'test_cuda.cpp',
            'test_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
