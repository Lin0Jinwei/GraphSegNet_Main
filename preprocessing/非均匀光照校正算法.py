import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import cupy as cp
import os
from datetime import datetime
import matplotlib
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'
matplotlib.use('TkAgg')  # 使用TkAgg后端，这个后端适用于生成图像文件但不显示它们
class fittingImageProcessor:
    def __init__(self, image_path, max_radius=100, min_radius=40, rgb_threshold=1, tolerance=1e-3, min_baseline_points=10):
        self.image_path = image_path
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.rgb_threshold = rgb_threshold
        self.tolerance = tolerance
        self.min_baseline_points = min_baseline_points
        self.image_rgb = cp.array(Image.open(image_path).convert("RGB"))
        self.base_regions = self.process_image()


    # 函数将平面拟合为圆内的像素
    def fit_plane(self, points):
        A = cp.c_[points[:, 0], points[:, 1], cp.ones(points.shape[0])]
        C, _, _, _ = cp.linalg.lstsq(A, points[:, 2], rcond=None)
        return C


    # 用于计算拟合的均方误差的函数
    def mse_plane(self, points, C):
        A = cp.c_[points[:, 0], points[:, 1], cp.ones(points.shape[0])]
        predicted = A @ C
        mse = cp.mean((predicted - points[:, 2]) ** 2)
        return mse


    # 获取圆内点的函数
    def get_circle_points(self, center, radius, image):
        x_center, y_center = center
        indices = cp.indices((image.shape[1], image.shape[0])).transpose(1, 2, 0).reshape(-1, 2)
        distances = cp.sqrt((indices[:, 0] - x_center) ** 2 + (indices[:, 1] - y_center) ** 2)
        mask = distances <= radius
        points = indices[mask]
        z_values = image[points[:, 1], points[:, 0]]
        return cp.c_[points, z_values]


    # 获取单像素厚度的圆环上的点
    def get_annulus_points(self, center, inner_radius, outer_radius, image):
        x_center, y_center = center
        indices = cp.indices((image.shape[1], image.shape[0])).transpose(1, 2, 0).reshape(-1, 2)
        distances = cp.sqrt((indices[:, 0] - x_center) ** 2 + (indices[:, 1] - y_center) ** 2)
        mask = (distances > inner_radius) & (distances <= outer_radius)
        points = indices[mask]
        z_values = image[points[:, 1], points[:, 0]]
        return cp.c_[points, z_values]


    # 用于判断外围像素RGB值与圆内像素RGB值变化的函数
    def is_significant_change(self, inner_points, outer_points, threshold):
        inner_mean = cp.mean(inner_points, axis=0)
        outer_mean = cp.mean(outer_points, axis=0)
        change = cp.linalg.norm(inner_mean - outer_mean)
        return change > threshold


    # 判断新圆是否与已有圆重叠
    def is_overlapping(self, new_center, new_radius, base_regions):
        x_new, y_new = new_center
        for x_center, y_center, radius in base_regions:
            distance = cp.sqrt((x_new - x_center) ** 2 + (y_new - y_center) ** 2)
            if distance < new_radius + radius:
                return True
        return False


    def process_image(self):
        height, width, _ = self.image_rgb.shape
        avg_rgb = cp.mean(self.image_rgb.reshape(-1, 3), axis=0)
        base_regions = []
        while len(base_regions) < self.min_baseline_points:
            x_center = random.randint(0, width - 1)
            y_center = random.randint(0, height - 1)
            center_rgb = self.image_rgb[y_center, x_center]
            rgb_diff = cp.linalg.norm(center_rgb - avg_rgb, ord=1)
            if rgb_diff > 50:
                continue
            radius = 5
            prev_C = None
            while radius <= self.max_radius:
                points = self.get_circle_points((x_center, y_center), radius, self.image_rgb)
                C = self.fit_plane(points)
                if prev_C is not None:
                    mse_change = self.mse_plane(points, C) - self.mse_plane(points, prev_C)
                    if mse_change > self.tolerance:
                        break
                if radius > 5:
                    inner_points = self.get_circle_points((x_center, y_center), radius - 5, self.image_rgb)
                    outer_points = self.get_annulus_points((x_center, y_center), radius - 5, radius, self.image_rgb)
                    if self.is_significant_change(inner_points[:, 2:], outer_points[:, 2:], self.rgb_threshold) and radius <= self.min_radius:
                        print("give up this point, now get " + str(len(base_regions)) + " point")
                        break
                    elif self.is_significant_change(inner_points[:, 2:], outer_points[:, 2:], self.rgb_threshold) and radius > self.min_radius:
                        if not self.is_overlapping((x_center, y_center), radius - 5, base_regions):
                            base_regions.append((x_center, y_center, radius - 5))
                            print("Currently available: " + str(len(base_regions)) + " point")
                        else:
                            break
                prev_C = C
                radius += 5
        return base_regions



    # 筛选出可能随机选在错误像素的圆圈
    def robust_mean(self, arrays, threshold=2.5):
        data = cp.array(arrays)
        medians = cp.median(data, axis=0)
        deviations = cp.abs(data - medians)
        mad = cp.median(deviations, axis=0)
        mad_std = mad * 1.4826
        is_outlier = deviations > (threshold * mad_std)
        data[is_outlier] = cp.nan
        means = cp.nanmean(data, axis=0)
        return means


    # 处理图片路径和背景色的RGB值
    def unify_background(self, image_path, background_rgb):
        image = Image.open(image_path)
        image_rgba = image.convert('RGBA')
        width, height = image.size
        for y in range(height):
            for x in range(width):
                r, g, b, a = image_rgba.getpixel((x, y))
                if abs(r - background_rgb[0]) < 5 and abs(g - background_rgb[1]) < 15 and abs(b - background_rgb[2]) < 15:
                    image_rgba.putpixel((x, y), (int(background_rgb[0]), int(background_rgb[1]), int(background_rgb[2]), a))
        image_rgb = image_rgba.convert('RGB')
        image_rgb.save('./fit_save/processed_image.png')
        # image_rgb.show()

    def extract_and_save_results(self):
        random.seed(42)  # 举例使用42作为种子，您可以选择任何数字
        save_path = './fit_save'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 加载图片
        image_path = self.image_path
        image_pil = Image.open(image_path).convert("RGB")
        image_rgb = cp.array(image_pil)

        # 处理图像并获取基线区域
        base_regions = self.base_regions
        print(base_regions)

        # 计算每个基本区域的平均颜色
        average_colors = []
        for x_center, y_center, radius in base_regions:
            points = self.get_circle_points((x_center, y_center), radius, image_rgb)
            average_color = cp.mean(points[:, 2:], axis=0)
            average_colors.append(average_color)

        # 显示结果
        print("圆圈的中心点坐标与圆圈的半径:", base_regions)
        print("圆圈的RGB三通道平均值:", average_colors)

        image_rgb_cpu = cp.asnumpy(image_rgb)
        base_regions_cpu = cp.asnumpy(base_regions)

        # 显示标记了基本区域的图像
        for x_center, y_center, radius in base_regions:
            cv2.circle(image_rgb_cpu, (x_center, y_center), radius, (255, 0, 0), 2)
        plt.imshow(image_rgb_cpu)
        now = datetime.now()
        date_time = now.strftime('%Y-%m-%d_%H-%M')
        plt.savefig(f"./fit_save/image_circle_{date_time}.png", dpi=1000, format='png')
        # plt.show()

        # 计算平均值
        result = self.robust_mean(average_colors)
        print("平均出的背景RGB值：", result)

        # 分离通道
        r, g, b = image_pil.split()
        # 将R通道作为灰度图像
        gray_image_r = r.convert("L")
        gray_image_g = g.convert("L")
        gray_image_b = b.convert("L")
        # 初始化列表来保存每个通道的坐标和像素值
        coordinate_R = []
        coordinate_G = []
        coordinate_B = []

        # 直接从给定的列表中获取圆圈的中心点坐标和对应的 RGB 三通道平均值
        average_color = []
        for x, y, radius in base_regions:
            # 获取 RGB 三通道平均值
            average_color = average_colors[base_regions.index((x, y, radius))]
            coordinate_R.append([x, y, round(average_color[0].item())])
            coordinate_G.append([x, y, round(average_color[1].item())])
            coordinate_B.append([x, y, round(average_color[2].item())])

        # 打印结果
        print("R通道坐标和像素值:", coordinate_R)
        print("G通道坐标和像素值:", coordinate_G)
        print("B通道坐标和像素值:", coordinate_B)

        # 将列表转换为 NumPy 数组
        points_R = cp.array(coordinate_R)
        points_G = cp.array(coordinate_G)
        points_B = cp.array(coordinate_B)

        # 为每个通道构建设计矩阵 A
        A_R = cp.c_[points_R[:, 0], points_R[:, 1], cp.ones(points_R.shape[0])]
        A_G = cp.c_[points_G[:, 0], points_G[:, 1], cp.ones(points_G.shape[0])]
        A_B = cp.c_[points_B[:, 0], points_B[:, 1], cp.ones(points_B.shape[0])]

        # 使用最小二乘法求解每个通道的平面拟合
        # lstsq 函数返回四个值：解、残差、秩和奇异值的数组
        C_R, _, _, _ = cp.linalg.lstsq(A_R, points_R[:, 2], rcond=None)
        C_G, _, _, _ = cp.linalg.lstsq(A_G, points_G[:, 2], rcond=None)
        C_B, _, _, _ = cp.linalg.lstsq(A_B, points_B[:, 2], rcond=None)

        # 解释结果
        print("R通道平面方程的系数为：A =", C_R[0], ", B =", C_R[1], ", C =", C_R[2])
        print("G通道平面方程的系数为：A =", C_G[0], ", B =", C_G[1], ", C =", C_G[2])
        print("B通道平面方程的系数为：A =", C_B[0], ", B =", C_B[1], ", C =", C_B[2])

        # 将通道转换为 NumPy 数组
        r_np = cp.array(r)
        g_np = cp.array(g)
        b_np = cp.array(b)

        # 创建一个新的图像数组来存储变换后的像素值
        transformed_r = cp.zeros_like(r_np)
        transformed_g = cp.zeros_like(g_np)
        transformed_b = cp.zeros_like(b_np)

        # 读取图像
        image = cv2.imread(image_path)
        image_gpu = cp.asarray(image)
        # 获取图像的高度和宽度
        height, width, _ = image_gpu.shape
        #
        # 生成索引矩阵
        i_matrix, j_matrix = cp.meshgrid(cp.arange(height), cp.arange(width), indexing='ij')

        # 创建一个新的图像数组来存储变换后平面拟合情况可视化像素值
        transformed_r_fitting = cp.zeros_like(r_np)
        transformed_g_fitting = cp.zeros_like(g_np)
        transformed_b_fitting = cp.zeros_like(b_np)

        # 计算变换后的像素值
        transformed_r_fitting = C_R[0] * i_matrix + C_R[1] * j_matrix + C_R[2]
        transformed_g_fitting = C_G[0] * i_matrix + C_G[1] * j_matrix + C_G[2]
        transformed_b_fitting = C_B[0] * i_matrix + C_B[1] * j_matrix + C_B[2]

        # 将 GPU 数组转换为 CPU 数组以保存为图像
        transformed_r_fitting_cpu = cp.asnumpy(transformed_r_fitting)
        transformed_g_fitting_cpu = cp.asnumpy(transformed_g_fitting)
        transformed_b_fitting_cpu = cp.asnumpy(transformed_b_fitting)

        # 将 NumPy 数组转换回 PIL 图像
        fitting_image_r_visible = Image.fromarray(transformed_r_fitting_cpu.astype('uint8'))
        fitting_image_g_visible = Image.fromarray(transformed_g_fitting_cpu.astype('uint8'))
        fitting_image_b_visible = Image.fromarray(transformed_b_fitting_cpu.astype('uint8'))

        # 合并通道
        fitting_image = Image.merge('RGB', (fitting_image_r_visible, fitting_image_g_visible, fitting_image_b_visible))
        # 显示图像
        # fitting_image.show()
        now = datetime.now()
        date_time = now.strftime('%Y-%m-%d_%H-%M')
        fitting_image.save(f"./fit_save/fitting_image_{date_time}.png")
        r, g, b = cp.asnumpy(result)
        bg_image = np.full((height, width, 3), (b, g, r), dtype=np.uint8) # 因为NumPy和OpenCV在处理图像时都使用BGR（蓝色、绿色、红色）颜色顺序
        cv2.imwrite(f'./fit_save/bg_image_{date_time}.jpg', bg_image)

        # 提取图像的各个通道
        B = image_gpu[:, :, 0]
        G = image_gpu[:, :, 1]
        R = image_gpu[:, :, 2]

        # 计算变换后的像素值
        transformed_r = cp.clip(C_R[0] * i_matrix + C_R[1] * j_matrix + R, 0, 255)
        transformed_g = cp.clip(C_G[0] * i_matrix + C_G[1] * j_matrix + G, 0, 255)
        transformed_b = cp.clip(C_B[0] * i_matrix + C_B[1] * j_matrix + B, 0, 255)

        # 将 GPU 数组转换为 CPU 数组以保存为图像
        transformed_r_cpu = cp.asnumpy(transformed_r)
        transformed_g_cpu = cp.asnumpy(transformed_g)
        transformed_b_cpu = cp.asnumpy(transformed_b)
        # 将 NumPy 数组转换回 PIL 图像
        transformed_r = Image.fromarray(transformed_r_cpu.astype('uint8'))
        transformed_g = Image.fromarray(transformed_g_cpu.astype('uint8'))
        transformed_b = Image.fromarray(transformed_b_cpu.astype('uint8'))

        # 分别保存三个通道的灰度图
        # 使用R通道创建灰度图
        gray_image_r = transformed_r
        gray_image_g = transformed_g
        gray_image_b = transformed_b
        # 保存灰度图
        gray_image_r.save('./fit_save/gray_image_r.jpg')
        gray_image_g.save('./fit_save/gray_image_g.jpg')
        gray_image_b.save('./fit_save/gray_image_b.jpg')

        # 合并通道
        transformed_image = Image.merge('RGB', (transformed_r, transformed_g, transformed_b))

        # 显示图像
        # transformed_image.show()
        now = datetime.now()
        date_time = now.strftime('%Y-%m-%d_%H-%M')
        transformed_image.save(f"./fit_save/transformed_image_{date_time}.png")

# 输入要校正展平的图片
image_processor = fittingImageProcessor('out_split_images/combined_image.png')
image_processor.extract_and_save_results()