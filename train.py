import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from PIL import Image
from matplotlib.widgets import RectangleSelector
from matplotlib import pyplot as plt
import time
import logging
import shutil
import json
from Canny import EdgeDetectionOptimizer
from model.segmentation_net import SegmentationNet

# 设置随机种子以确保实验可重复性
np.random.seed(44)


class ImageSegmentation:
    """
    图像分割类：用于训练和评估图像分割模型。
    """

    def __init__(self, input_image_path, max_iter=100, min_labels=3, learning_rate=0.2,
                 num_channels=20, num_convs=2, image_quality=1, use_cuda=False, stable_label_count=20,
                 mode='train', model_weight=''):
        self.input_image_path = input_image_path
        self.output_folder_path = self._create_output_folder(f"./train_results/train_results_{time.strftime('%Y%m%d_%H%M%S')}")
        self.use_cuda = use_cuda
        self.max_iter = max_iter
        self.min_labels = min_labels
        self.learning_rate = learning_rate
        self.visualize = True
        self.num_channels = num_channels
        self.num_convs = num_convs
        self.label_colors = np.random.randint(255, size=(100, 3))  # 随机颜色映射
        self.target_rgb_image = []
        self.image_quality = image_quality
        self.stable_label_count = stable_label_count
        self.mode = mode
        self.similarity_step_size = 0.5
        self.continuity_step_size = 2
        self.model_weight = model_weight
        self._setup_logging()

    def _create_output_folder(self, folder_name):
        """创建输出文件夹"""
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name

    def _setup_logging(self):
        """设置日志记录"""
        log_file = os.path.join(self.output_folder_path, "training_log.txt")
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # 复制当前脚本到输出文件夹
        script_path = os.path.abspath(__file__)
        shutil.copyfile(script_path, os.path.join(self.output_folder_path, os.path.basename(script_path)))

    def _resize_image(self, input_path):
        """调整图像大小"""
        with Image.open(input_path) as img:
            original_size = img.size
            new_size = (original_size[0] // self.image_quality, original_size[1] // self.image_quality)
            resized_img = img.resize(new_size, Image.LANCZOS)
            base, ext = os.path.splitext(os.path.basename(self.input_image_path))
            output_resize_path = os.path.join(self.output_folder_path, f'{base}_resize{ext}')
            resized_img.save(output_resize_path)
        return resized_img

    def _read_image(self):
        """读取并预处理图像"""
        self.image = self._resize_image(self.input_image_path)
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
        self.red_channel, self.green_channel, self.blue_channel = self.image.split()

        # 保存各通道图像
        self.image.save(os.path.join(self.output_folder_path, 'original_image.jpeg'))
        self.red_channel.save(os.path.join(self.output_folder_path, 'red_channel.jpeg'))
        self.green_channel.save(os.path.join(self.output_folder_path, 'green_channel.jpeg'))
        self.blue_channel.save(os.path.join(self.output_folder_path, 'blue_channel.jpeg'))

    def _load_image(self):
        """加载图像并转换为模型输入格式"""
        self.input_image = 'red_channel.jpeg'
        self.image_data = cv2.imread(os.path.join(self.output_folder_path, self.input_image))
        self.tensor_data = torch.from_numpy(np.array([self.image_data.transpose((2, 0, 1)).astype('float32') / 255.]))
        if self.use_cuda:
            self.tensor_data = self.tensor_data.cuda()
        self.tensor_data = Variable(self.tensor_data)

    def _define_model(self):
        """定义模型"""
        self.model = SegmentationNet(self.tensor_data.size(1), self.num_channels, self.num_convs)
        if self.use_cuda:
            self.model.cuda()
        if self.mode == 'train':
            self.model.train()
        else:
            self.model.load_state_dict(torch.load(self.model_weight))

    def _train_model(self):
        """训练模型"""
        cross_entropy_loss = nn.CrossEntropyLoss()
        smooth_l1_loss = nn.SmoothL1Loss()
        l1_loss = nn.L1Loss(size_average=True)

        # 初始化目标张量
        hpy_target = torch.zeros(self.image_data.shape[0] - 1, self.image_data.shape[1], self.num_channels)
        hpz_target = torch.zeros(self.image_data.shape[0], self.image_data.shape[1] - 1, self.num_channels)
        if self.use_cuda:
            hpy_target = hpy_target.cuda()
            hpz_target = hpz_target.cuda()

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        stable_count = 0
        prev_num_labels = 0

        for batch_idx in range(self.max_iter):
            optimizer.zero_grad()
            output = self.model(self.tensor_data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, self.num_channels)
            output_hp = output.reshape((self.image_data.shape[0], self.image_data.shape[1], self.num_channels))
            hpy = output_hp[1:, :, :] - output_hp[:-1, :, :]
            hpz = output_hp[:, 1:, :] - output_hp[:, :-1, :]
            lhpy = l1_loss(hpy, hpy_target)
            lhpz = l1_loss(hpz, hpz_target)

            _, target = torch.max(output, 1)
            target_np = target.data.cpu().numpy()
            num_labels = len(np.unique(target_np))

            if num_labels == prev_num_labels:
                stable_count += 1
            else:
                stable_count = 0
                prev_num_labels = num_labels

            if self.visualize:
                self.target_rgb_image = np.array([self.label_colors[c % self.num_channels] for c in target_np])
                self.target_rgb_image = self.target_rgb_image.reshape(self.image_data.shape).astype(np.uint8)
                resized_image = cv2.resize(self.target_rgb_image, (1080, 720))
                cv2.imshow("output", resized_image)
                if num_labels != prev_num_labels:
                    output_image_path = os.path.join(self.output_folder_path, f"output_batch{batch_idx}_labels{num_labels}.png")
                    cv2.imwrite(output_image_path, self.target_rgb_image)
                    prev_num_labels = num_labels
                cv2.waitKey(10)

            # 动态损失权重调整策略
            if stable_count > 5:
                self.continuity_step_size *= 1.1
            elif stable_count < 3:
                self.continuity_step_size *= 0.9

            target_one_hot = torch.nn.functional.one_hot(target, num_classes=20).float()
            loss = smooth_l1_loss(output, target_one_hot) + \
                   self.similarity_step_size * cross_entropy_loss(output, target) + \
                   self.continuity_step_size * (lhpy + lhpz)
            loss.backward()
            optimizer.step()
            logging.info(f"Batch {batch_idx}/{self.max_iter} | Labels: {num_labels} | Loss: {loss.item()}")

            if num_labels <= self.min_labels:
                logging.info(f"Labels {num_labels} reached minimum {self.min_labels}.")
                break
            if stable_count >= self.stable_label_count:
                logging.info("Labels have been stable for multiple iterations.")
                break

        self._save_model(self.output_folder_path)
        self._save_variables(self.output_folder_path)

    def _save_model(self, path):
        """保存模型权重"""
        torch.save(self.model.state_dict(), os.path.join(path, 'model_weights.pth'))

    def _save_variables(self, path):
        """保存训练变量"""
        variables = {
            'input_image_path': self.input_image_path,
            'output_folder_path': self.output_folder_path,
            'use_cuda': self.use_cuda,
            'max_iter': self.max_iter,
            'min_labels': self.min_labels,
            'learning_rate': self.learning_rate,
            'num_channels': self.num_channels,
            'num_convs': self.num_convs,
            'image_quality': self.image_quality,
            'stable_label_count': self.stable_label_count,
            'mode': self.mode,
            'similarity_step_size': self.similarity_step_size,
            'continuity_step_size': self.continuity_step_size,
        }
        with open(os.path.join(path, 'variables.json'), 'w', encoding='utf-8') as f:
            json.dump(variables, f, indent=4, ensure_ascii=False)

    def run(self):
        """运行图像分割流程"""
        self._read_image()
        self._load_image()
        self._define_model()
        self._train_model()
        self._save_results()
        self._post_process()

    def _save_results(self):
        """保存分割结果"""
        base_name = os.path.splitext(os.path.basename(self.input_image_path))[0]
        output_file = os.path.join(self.output_folder_path, f"{base_name}_result.png")
        cv2.imwrite(output_file, self.target_rgb_image)
        logging.info(f"Results saved to {output_file}")

    def _post_process(self):
        """后处理：边缘检测优化"""
        base_name = os.path.splitext(os.path.basename(self.input_image_path))[0]
        output_file = os.path.join(self.output_folder_path, f"{base_name}_result.png")
        optimizer = EdgeDetectionOptimizer(self.input_image_path, output_file, self.output_folder_path)
        optimizer.run()


if __name__ == '__main__':
    segmentor = ImageSegmentation(
        input_image_path=r'datasets/training_images/cropped_resized_34_original.png',
        max_iter=100,
        min_labels=3,
        learning_rate=0.1,
        num_channels=20,
        num_convs=2,
        image_quality=1,
        use_cuda=False,
        mode='train',
        model_weight='./train_results/train_results_20250317_174918/model_weights.pth'
    )
    segmentor.run()