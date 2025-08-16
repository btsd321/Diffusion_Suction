# SuctionNet-1Billion 基线方法

本文档对应 RA-L 论文 "SuctionNet-1Billion:  A  Large-Scale  Benchmark  for  Suction  Grasping" 中的基线方法。

![框架图](https://github.com/graspnet/suctionnet-baseline/blob/master/framework3.jpg)

## 数据集

请从我们的 [SuctionNet 官网](https://graspnet.net/suction) 下载数据和标签。

## 环境配置

本代码在 `CUDA 10.1` 和 `pytorch 1.4.0`，操作系统为 ubuntu `16.04` 下测试通过。

## 训练前的准备工作

训练网络前，需要额外的标签，包括密封标签的二维映射、物体的包围盒和物体中心点。

请先切换到 `neural_network` 目录：

```
cd neural_network
```

生成吸取分数热力图，请运行如下命令：

```
python score_mapping.py \
--data_root /home/lixinlong/Project/pose_detect_train/Data/GraspNet \
--saveroot /home/lixinlong/Project/pose_detect_train/Data/GraspNet/additional_label \
--camera realsense \
--sigma 4 \
--pool_size 10 \
--start_scene_idx 0 \
--end_scene_idx 99 \
--save_visu
```

或者修改 [scripts/score_mapping.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/score_mapping.sh) 并运行 `sh scripts/score_mapping.sh`。

获取物体包围盒和中心点，请运行如下命令：

```
python cal_center_bbox.py \
--data_root /home/lixinlong/Project/pose_detect_train/Data/GraspNet \
--saveroot /home/lixinlong/Project/pose_detect_train/Data/GraspNet/additional_label \
--camera realsense \
--pool_size 10 \
--start_scene_idx 00 \
--end_scene_idx 99 \
--save_visu
```

或者修改 [scripts/cal_center_bbox.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/cal_center_bbox.sh) 并运行 `sh scripts/cal_center_bbox.sh`。

请确保上述两条命令的 `--saveroot` 参数一致。

注意：密封标签的二维映射文件可能会占用高达 `177 G` 的磁盘空间。我们建议提前生成并保存，以提升训练效率。你也可以将其修改为在线生成，但训练速度会大幅降低。

## 使用方法

### 神经网络训练与推理

切换到 `neural_network` 目录：

```
cd neural_network
```

训练模型，请使用如下命令：

```
python train.py \
--model model_name \ 
--camera realsense \ # 选择 realsense 或 kinect
--log_dir /path/to/save/the/model/weights \
--data_root /path/to/SuctionNet/dataset \
--label_root /path/to/the/additional/labels \
--batch_size 8
```

或者修改 [scripts/deeplabv3plus_train.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_train.sh)、[scripts/deeplabv3plus_train_depth.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_inference_depth.sh)、[scripts/convnet_train.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/convnet_train.sh) 来训练我们的 RGB-D 模型、深度模型和全卷积网络(FCN)模型。

推理时，请使用如下命令：

```
python inference.py \
--model model_name \
--checkpoint_path /path/to/the/saved/model/weights \
--split test_seen \ # 可选 test, test_seen, test_similar, test_novel
--camera realsense \ # 选择 realsense 或 kinect
--dataset_root /path/to/SuctionNet/dataset \
--save_dir /path/to/save/the/inference/results \
--save_visu # 是否保存可视化结果
```

或者修改 [scripts/deeplabv3plus_inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_inference.sh)、[scripts/deeplabv3plus_inference_depth.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_inference_depth.sh)、[scripts/convnet_inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/convnet_inference.sh) 进行推理。

### Normal STD(法向标准差方法)

切换到 `normal_std` 目录：

```
cd normal_std
```

该方法无需训练，可直接推理，命令如下：

```
python inference.py 
--split test_seen \ # 可选 test, test_seen, test_similar, test_novel
--camera realsense \ # 选择 realsense 或 kinect
--save_root /path/to/save/the/inference/results \
--dataset_root /path/to/SuctionNet/dataset \
--save_visu
```

或者修改 [inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/normal_std/inference.sh) 并运行 `sh inference.sh`。

## 预训练模型

### RGB-D 模型

我们提供了以下模型：[realsense 预训练模型](https://drive.google.com/file/d/18TbctdhpNXEKLYDWFzI9cT1Wnhe-tn9h/view?usp=sharing)、[kinect 预训练模型](https://drive.google.com/file/d/1gOz_KmIugBGUtpcyHAgYO01T0h5ZqOl9/view?usp=sharing)、[Fully Conv Net for realsense](https://drive.google.com/file/d/1hgYYIvw5Xy-r5C8IitKizswtuMV_EqPP/view?usp=sharing)、[Fully Conv Net for kinect](https://drive.google.com/file/d/1A6K5EmItBuDaxrWyz5g8zSHY5Kw1_NnX/view?usp=sharing)。

### 深度模型

仅使用深度图的模型也已提供：[realsense 深度模型](https://drive.google.com/file/d/1q2W2AV663PNT4_TYo5zZtYxjenZJ7GAb/view?usp=sharing) 和 [kinect 深度模型](https://drive.google.com/file/d/1mAzFC9dlEDBuoHQp7JGTcTkKGSwFnVth/view?usp=sharing)。

## 论文引用

如果本项目对您的研究有帮助，请引用如下论文：

```
@ARTICLE{suctionnet,
  author={Cao, Hanwen and Fang, Hao-Shu and Liu, Wenhai and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters}, 
  title={SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping}, 
  year={2021},
  volume={6},
  number={4},
  pages={8718-8725},
  doi={10.1109/LRA.2021.3115406}}
```