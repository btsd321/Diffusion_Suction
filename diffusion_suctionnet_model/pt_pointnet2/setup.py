from setuptools import setup, find_packages

setup(
    name="pt_pointnet2",
    version="0.1.0",
    description="PyTorch implementation of PointNet++ sampling and utilities.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch >= 2.0.0"
    ],
    python_requires='>=3.7',
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
