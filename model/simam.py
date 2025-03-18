import torch.nn as nn

class SimAMModule(nn.Module):
    """
    SimAM 注意力模块：用于增强特征图的表达能力。
    """

    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAMModule, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        return f"{self.__class__.__name__}(lambda={self.e_lambda})"

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        num_pixels = height * width - 1

        # 计算特征图的均值和方差
        mean = x.mean(dim=[2, 3], keepdim=True)
        variance = (x - mean).pow(2)
        y = variance / (4 * (variance.sum(dim=[2, 3], keepdim=True) / num_pixels + self.e_lambda) + 0.5)

        return x * self.activation(y)