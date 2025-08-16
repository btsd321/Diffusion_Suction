import os
import glob
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# 获取当前文件所在目录
this_dir = os.path.dirname(os.path.abspath(__file__))

# 读取版本信息
def read_version():
    version_file = os.path.join(this_dir, 'diffusion_suctionnet_model', '_version.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            exec(f.read())
            return locals()['__version__']
    return '0.1.0'

# 读取 README 文件
def read_readme():
    readme_path = os.path.join(this_dir, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# 读取依赖
def read_requirements():
    req_path = os.path.join(this_dir, 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    return [
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'tqdm>=4.50.0',
        'h5py>=2.10.0',
    ]

# 配置 PointNet2 CUDA 扩展
def get_extensions():
    extensions = []
    
    # PointNet2 扩展
    pointnet2_dir = os.path.join(this_dir, 'pointnet2')
    if os.path.exists(pointnet2_dir):
        ext_src_root = os.path.join(pointnet2_dir, "_ext_src")
        if os.path.exists(ext_src_root):
            ext_sources = glob.glob(os.path.join(ext_src_root, "src", "*.cpp")) + \
                         glob.glob(os.path.join(ext_src_root, "src", "*.cu"))
            
            if ext_sources:
                extensions.append(
                    CUDAExtension(
                        name="diffusion_suctionnet_model.pointnet2._ext",
                        sources=ext_sources,
                        extra_compile_args={
                            "cxx": ["-O3"],
                            "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
                        },
                        include_dirs=[os.path.join(ext_src_root, "include")],
                    )
                )
    
    return extensions

# 包配置
setup(
    name="diffusion_suctionnet_model",
    version=read_version(),
    author="btsd321",
    author_email="",
    description="Diffusion SuctionNet Model - A PyTorch implementation for 6DoF suction grasping",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/btsd321/diffusion_suctionnet_model",
    packages=find_packages(),
    package_data={
        'diffusion_suctionnet_model': [
            'diffusers/schedulers/*.py',
            'data/*.py',
            'utils/*.py',
            'pointnet2/*.py',
            'pointnet2/_ext_src/src/*.cpp',
            'pointnet2/_ext_src/src/*.cu',
            'pointnet2/_ext_src/include/*.h',
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=read_requirements(),
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension,
    },
    zip_safe=False,
)
