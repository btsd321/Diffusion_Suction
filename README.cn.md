## Diffusion Suction Grasping with Large-Scale Parcel Dataset
## 基于大规模包裹数据集的扩散吸取抓取

扩散-吸取-去噪过程示意图。  
Illustration of the  suction-diffusion-denoising  process.
![Alt text](/images/1.gif)

本项目为论文 [**Diffusion Suction Grasping with Large-Scale Parcel Dataset**] 的 PyTorch 版本代码。

## Diffusion-Suction 架构概览
用于堆叠场景下 6DoF 位姿估计的 Diffusion-Suction 架构示意图。  
Illustration of the Diffusion-Suction architecture for 6DoF Pose Estimation in stacked scenarios.
![Alt text](/images/model1.png)

## 包裹吸取数据集概览
自监督包裹吸取标签生成流程示意图。  
Illustration of the Self-Parcel-Suction-Labeling pipeline.
![Alt text](/images/model2.png)

## 定性结果展示
SuctionNet-1Billion 数据集上的评估结果  
Evaluation SuctionNet-1Billion dataset
![Alt text](/images/dataset1.png)
Parcel-Suction-Dataset 数据集上的评估结果  
Evaluation Parcel-Suction-Dataset dataset
![Alt text](/images/dataset2.png)

---

## 快速开始

### 1. 环境准备
请先克隆本仓库到本地：
```bash
git clone https://github.com/TAO-TAO-TAO-TAO-TAO/Diffusion_Suction.git
```
安装环境依赖：

- 安装 [Pytorch](https://pytorch.org/get-started/locally/)。本项目需使用 GPU，代码在 Ubuntu 16.04/18.04、CUDA 10.0 和 cuDNN v7.4、python3.6 下测试通过。
- 主干网络 PointNet++ 源自 [pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch)。
- 编译 PointNet++ 的 CUDA 层（用于主干网络）：

    ```bash
    cd train\Sparepart\train.py
    python train.py install
    ```

- 安装以下 Python 依赖（使用 `pip install`）：

    ```bash
    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'
    torch==1.1.0
    torchvision==0.3.0
    sklearn
    h5py
    nibabel
    ```

---

### 2. 训练 Diffusion-Suction 网络
```bash
cd train\Sparepart\train.py
python train.py 
```

---

### 3. 在自定义数据上评估

- Parcel-Suction-Dataset 可在 [这里](https://drive.google.com/drive/folders/1l4jz7LE7HXdn2evylodggReTTnip7J1Q?usp=sharing) 下载。
- SuctionNet-1Billion 可在 [这里](https://github.com/graspnet/suctionnetAPI) 获取。

#### 评估指标
评估指标的 Python 代码可在 [这里](https://github.com/graspnet/suctionnetAPI) 获取。

---

## 论文引用
如果本项目对您的研究有帮助，请引用以下论文：

```bibtex
@article{huang2025diffusion,
title={Diffusion Suction Grasping with Large-Scale Parcel Dataset},
author={Huang, Ding-Tao and He, Xinyi and Hua, Debei and Yu, Dongfang and Lin, En-Te and Zeng, Long},
journal={arXiv preprint arXiv:2502.07238},
year={2025}
}

@inproceedings{huang2025diffusion,
title={Diffusion Suction Grasping with Large-Scale Parcel Dataset},
author={dingtao huang, Debei Hua, Dongfang Yu, Xinyi He, Ente Lin, lianghong wang, Jinliang Hou, Long Zeng},
booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
year={2025},
organization={IEEE}
}
```

---

## 联系方式

如有任何问题，欢迎联系作者：

Ding-Tao Huang: [hdt22@mails.tsinghua.edu.cn](hdt22@mails.tsinghua.edu.cn)

