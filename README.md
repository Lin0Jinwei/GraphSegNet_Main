# GraphSegNet: 基于卷积神经网络的二维石墨烯图像无监督分割框架
---

## 📂 代码结构

```plaintext
GraphSegNet_Main/
├── model/                          # 模型定义
│   ├── segmentation_net.py         # 网络架构
│   └── simam.py                    # SimAM注意力模块
│
├── preprocessing/                  # 数据预处理工具
│   ├── illumination_correction.py  # 非均匀光照校正算法
│   └── darkfield_denoising.py      # 暗场去噪算法
│
├── datasets/                       # 数据集
│   ├── training_images/            # 训练图像
│   └── testing_images/             # 测试图像
│       ├── 100x/                   # 100倍显微图像
│       └── 500x/                   # 500倍显微图像
│
├── utils/                          # 工具函数
│   ├── canny.py                    # Canny边缘检测
│   └── canny_parameter_search.py   # 贝叶斯优化Canny参数
├── ablation/                       # 消融实验
├── train.py                        # 训练脚本
├── test.py                         # 测试脚本
└── evaluate.py                     # 指标评估（mIoU/SSIM）
