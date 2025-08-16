#!/bin/bash

# 构建和安装 diffusion_suctionnet_model 包

set -e

echo "构建 diffusion_suctionnet_model 包..."

# 清理之前的构建
echo "清理旧构建文件..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 检查依赖
echo "检查构建依赖..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import setuptools; print(f'setuptools version: {setuptools.__version__}')"

# 构建包
echo "构建 Python 包..."
python setup.py sdist bdist_wheel

# 安装包（开发模式）
echo "安装包到本地环境（开发模式）..."
pip install -e .

# 验证安装
echo "验证安装..."
python -c "import diffusion_suctionnet_model; print(f'Package version: {diffusion_suctionnet_model.__version__}')"

echo "构建完成！"
echo "包文件位于 dist/ 目录中"
echo "要发布到 PyPI，请运行："
echo "  twine upload dist/*"
