import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import glob
import datetime
import tqdm
from torch.autograd import Variable
from PIL import Image
from collections import Counter
from model.segmentation_net import SegmentationNet

# 配置
use_cuda = False  # 是否使用CUDA
np.random.seed(44)  # 设置随机种子确保结果可重复

# 模型权重路径
weights_path = r'train_results\train_results_20250318_200730\model_weights.pth'

nChannel = 20

# 获取图像中最常见的颜色
def get_most_common_color(image):
    pixels = list(image.getdata())
    color_counter = Counter(pixels)
    return color_counter.most_common(1)[0][0]

# 去除图像边缘噪声
def remove_noise(image_path, output_dir):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # 转换为 NumPy 数组
    width, height = img.size

    # 获取背景颜色
    pixels = img_np.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    background_color = colors[np.argmax(counts)]

    # 设置边缘区域为背景色
    border_size = 2
    img_np[:border_size, :] = background_color  # 上边缘
    img_np[-border_size:, :] = background_color  # 下边缘
    img_np[:, :border_size] = background_color  # 左边缘
    img_np[:, -border_size:] = background_color  # 右边缘

    # 保存结果
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    Image.fromarray(img_np).save(output_path)

# 自定义固定颜色映射
def get_color_map(n_classes):
    color_map = np.array([
        [0, 0, 255],  # 蓝色
        [255, 255, 0],  # 黄色
        [0, 255, 255],  # 青色
        [255, 165, 0],  # 橙色
        [34, 139, 34],  # 森林绿
    ])
    if n_classes > len(color_map):
        color_map = np.tile(color_map, (n_classes // len(color_map) + 1, 1))[:n_classes]
    return color_map

# 多文件夹处理
def process_images_in_folder(input_folder, output_parent_folder):
    # 确保输出父文件夹存在
    if not os.path.exists(output_parent_folder):
        os.makedirs(output_parent_folder)

    # 创建一个基于当前时间戳的新文件夹以避免重复
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_output_folder = os.path.join(output_parent_folder, f'test_{timestamp}')
    os.makedirs(current_output_folder, exist_ok=True)

    # 获取颜色映射
    color_map = get_color_map(nChannel)

    # 遍历输入文件夹中的所有子文件夹
    for subdir in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir)
        if os.path.isdir(subdir_path):
            result_folder = os.path.join(current_output_folder, f'{subdir}_result')
            os.makedirs(result_folder, exist_ok=True)
            original_images_folder = result_folder
            predicted_images_folder = result_folder
            print(f'Processing images in folder: {subdir}')

            # 获取该子文件夹中的所有图像
            img_list = sorted(glob.glob(os.path.join(subdir_path, '*')))
            if len(img_list) == 0:
                print(f"No images found in the folder: {subdir_path}")
                continue

            # 加载模型
            model = SegmentationNet(input_dim=3, num_channels=nChannel, num_convs=2)
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            model = model.cuda() if use_cuda else model

            for img_file in tqdm.tqdm(img_list):
                im = cv2.imread(img_file)
                if im is None:
                    print(f"Failed to load image: {img_file}. Skipping.")
                    continue

                # 调整图像大小和转换为适合输入模型的格式
                data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
                data = data.cuda() if use_cuda else data
                data = Variable(data)
                output = model(data)[0]
                output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
                ignore, target = torch.max(output, 1)
                inds = target.data.cpu().numpy().reshape((im.shape[0], im.shape[1]))

                # 使用固定颜色映射
                inds_rgb = color_map[inds]

                # 保存原图和预测图
                base_filename = os.path.splitext(os.path.basename(img_file))[0]
                original_output_filepath = os.path.join(original_images_folder, f'{base_filename}_original.png')
                predicted_output_filepath = os.path.join(predicted_images_folder, f'{base_filename}_predicted.png')

                # 保存原图
                cv2.imwrite(original_output_filepath, im)

                # 保存预测图
                cv2.imwrite(predicted_output_filepath, inds_rgb)
                remove_noise(predicted_output_filepath, predicted_images_folder)

# 设置输入和输出路径
input_directory = './datasets/testing_images/500x/'  # 输入图像路径
output_directory = './test_result'  # 输出结果路径
process_images_in_folder(input_directory, output_directory)
