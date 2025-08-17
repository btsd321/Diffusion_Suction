#!/bin/bash

# 扩散吸取网络推理脚本
# 对应VSCode调试配置的非调试版本，用于准确的性能测试

echo "=========================================="
echo "扩散吸取网络推理 - 性能测试模式"
echo "=========================================="

# 设置工作目录
cd /home/lixinlong/Project/pose_detect_train/example

# 检查GPU状态
echo "GPU状态检查:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

echo ""
echo "开始推理..."

conda activate linux_conda

# 运行推理（非调试模式）
python3 inference.py \
    --checkpoint "/home/lixinlong/Project/pose_detect_train/example/input/model/20250720/checkpoint.tar" \
    --rgb "/home/lixinlong/Project/pose_detect_train/example/input/rgb_images/test_bgr_30-30.png" \
    --depth "/home/lixinlong/Project/pose_detect_train/example/input/depth_images/test_depth_30-30.png" \
    --mask "/home/lixinlong/Project/pose_detect_train/example/input/segment_single_images/test_segment_30-30-26.png" \
    --camera_info "/home/lixinlong/Project/pose_detect_train/example/input/camera_info.yaml" \
    --params "/home/lixinlong/Project/pose_detect_train/example/input/parameter.json" \
    --enable_profiling \
    --diffusion_steps 20 \
    --output "./inference_results_production.npz"

echo ""
echo "推理完成！"

# 显示结果文件
echo "生成的文件:"
ls -la inference_results_production*

echo ""
echo "性能对比建议:"
echo "1. 比较调试模式 vs 生产模式的 inference_time"
echo "2. 调试模式通常会慢 2-5 倍"
echo "3. 生产部署时请使用非调试模式"
