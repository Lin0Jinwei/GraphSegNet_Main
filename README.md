基于卷积神经网络的二维石墨   烯图像无监督分割方法研究
Unsupervised segmentation of 2D graphene images based on convolutional neural networks

代码结构：
GraphSegNet_Main/
├── model/
│   ├── segmentation_net.py    # 网络架构
│   └── simam.py               # SimAM注意力模块
├── preprocessing/             # 数据预处理
│   ├── 非均匀光照校正算法.py    # 光照校正
│   └── 暗场去噪算法.py         # 噪声抑制
├── datasets/                  # 数据集
│        ├── training_images/
│        └── testing_images/
│            ├── 100x/
│            └── 500x/
├── utils/                     # 工具函数
│      ├── canny.py            # canny边缘检测
│      └── cannyOptimalParameterSearch.py  # 贝叶斯搜索canny参数
├── train.py                   # 训练脚本
├── test.py                    # 测试脚本
└── evaluate.py                # 指标评估（mIoU/SSIM）
