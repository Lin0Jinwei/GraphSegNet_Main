from skopt import gp_minimize
from skopt.space import Integer, Categorical
import cv2
import numpy as np
import os


class EdgeDetectionOptimizer:
    def __init__(self, img_path1, img_path2, output_dir='output_images', best_params = {
        'blur_kernel': 3,  # 最优模糊核大小
        'canny_low_threshold': 50,  # 最优Canny低阈值
        'canny_high_threshold': 90,  # 最优Canny高阈值
        'morphology_kernel_size': 5,  # 最优形态学操作核大小
        'dilate_iterations': 3  # 最优膨胀迭代次数
    }):
        """
        初始化边缘检测优化器，使用固定的最佳参数

        Parameters:
        - img_path1: 原图路径 (未进行任何处理)
        - img_path2: 分割后的预测图路径
        - output_dir: 保存结果的目录
        - best_params: 最优参数字典 (如果为 None，则进行优化)
        """
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.output_dir = output_dir
        self.best_params = best_params

        # 读取图像
        self.img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        self.img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 定义边缘检测算法
        self.edge_detectors = {
            'Canny': lambda img, params: cv2.Canny(cv2.GaussianBlur(img, (params[0], params[0]), 0), params[1],
                                                   params[2]),
        }

        if not self.best_params:
            # 定义搜索空间
            self.search_space = [
                Categorical([3, 5, 7], name='blur_kernel'),
                Integer(10, 50, name='canny_low_threshold'),
                Integer(90, 200, name='canny_high_threshold'),
                Integer(3, 5, name='morphology_kernel_size'),
                Integer(1, 3, name='dilate_iterations')
            ]
        else:
            self.search_space = []

    def calculate_iou(self, mask_true, mask_pred):
        """
        计算两个二值图像之间的交并比 (IoU)

        Parameters:
        - mask_true: 真值掩码
        - mask_pred: 预测掩码

        Returns:
        - IoU值
        """
        mask_true_bool = mask_true.astype(bool)
        mask_pred_bool = mask_pred.astype(bool)
        intersection = np.logical_and(mask_true_bool, mask_pred_bool).sum()
        union = np.logical_or(mask_true_bool, mask_pred_bool).sum()
        if union == 0:
            return 0
        iou = intersection / union
        return iou

    def objective(self, params, edge_detector_name):
        """
        优化目标函数，计算IoU的负值

        Parameters:
        - params: 当前参数组合
        - edge_detector_name: 使用的边缘检测算法名称

        Returns:
        - 负的IoU值，用于最小化优化
        """
        edge_detector = self.edge_detectors[edge_detector_name]

        edges_true = edge_detector(self.img1, params)
        kernel = np.ones((params[3], params[3]), np.uint8)
        filled_edges_true = cv2.morphologyEx(edges_true, cv2.MORPH_CLOSE, kernel)
        filled_edges_true = cv2.dilate(filled_edges_true, kernel, iterations=params[4])

        edges_pred = edge_detector(self.img2, params)
        filled_edges_pred = cv2.morphologyEx(edges_pred, cv2.MORPH_CLOSE, kernel)
        filled_edges_pred = cv2.dilate(filled_edges_pred, kernel, iterations=params[4])

        iou = self.calculate_iou(filled_edges_true, filled_edges_pred)
        return -iou

    def optimize(self):
        """
        执行边缘检测优化过程，寻找最佳参数组合
        """
        if not self.best_params:
            # 之前的优化代码
            ...
        else:
            # 如果已有固定参数，直接使用这些参数
            print("Using fixed parameters for edge detection...")
            results = {}
            for detector in self.edge_detectors:
                print(f"Running for {detector} using fixed parameters...")
                edge_detector = self.edge_detectors[detector]

                # 使用最佳固定参数进行边缘检测
                params = self.best_params
                edges_true_best = edge_detector(self.img1, [params['blur_kernel'], params['canny_low_threshold'],
                                                            params['canny_high_threshold']])
                kernel = np.ones((params['morphology_kernel_size'], params['morphology_kernel_size']), np.uint8)
                filled_edges_true_best = cv2.morphologyEx(edges_true_best, cv2.MORPH_CLOSE, kernel)
                filled_edges_true_best = cv2.dilate(filled_edges_true_best, kernel,
                                                    iterations=params['dilate_iterations'])

                edges_pred_best = edge_detector(self.img2, [params['blur_kernel'], params['canny_low_threshold'],
                                                            params['canny_high_threshold']])
                filled_edges_pred_best = cv2.morphologyEx(edges_pred_best, cv2.MORPH_CLOSE, kernel)
                filled_edges_pred_best = cv2.dilate(filled_edges_pred_best, kernel,
                                                    iterations=params['dilate_iterations'])

                final_iou = self.calculate_iou(filled_edges_true_best, filled_edges_pred_best)
                print(f"Final IoU after optimization for {detector}: {final_iou:.4f}")

                # 保存结果图像
                cv2.imwrite(os.path.join(self.output_dir, f'filled_edges_true_best_{detector}.png'),
                            filled_edges_true_best)
                cv2.imwrite(os.path.join(self.output_dir, f'filled_edges_pred_best_{detector}.png'),
                            filled_edges_pred_best)

            print("Final results saved to:", self.output_dir)

            return results

    def save_iou_history(self, iou_history):
        """
        将IoU历史保存到文本文件中

        Parameters:
        - iou_history: 记录了每次优化迭代中IoU值的列表
        """
        iou_history_path = os.path.join(self.output_dir, 'iou_history.txt')
        with open(iou_history_path, 'w') as f:
            for entry in iou_history:
                f.write(entry + '\n')
        print(f"IoU history saved to: {iou_history_path}")

    def save_results(self, results):
        """
        保存优化结果

        Parameters:
        - results: 结果字典，包含每个边缘检测方法的最佳参数和IoU值
        """
        # 这里保存结果的方式可以根据您的需求进行修改
        with open(os.path.join(self.output_dir, 'optimization_results.txt'), 'w') as f:
            for detector, result in results.items():
                f.write(f"Detector: {detector}\n")
                f.write(f"Best Params: {result['best_params']}\n")
                f.write(f"Final IoU: {result['iou']:.4f}\n")
                f.write('-' * 40 + '\n')

        print(f"Results saved to {os.path.join(self.output_dir, 'optimization_results.txt')}")

    def run(self):
        """
        执行整个优化流程，优化参数并保存结果
        """
        results = self.optimize()
        self.save_results(results)


if __name__ == '__main__':
    # Example usage with fixed parameters
    img_path1 = 'original_images/maskouto_merge_graphene_2020723_0002_C.png'
    img_path2 = 'predicted_images/maskouto_merge_graphene_2020723_0002_C.png'

    # Use fixed best parameters for edge detection
    best_params = {
        'blur_kernel': 3,  # 最优模糊核大小
        'canny_low_threshold': 50,  # 最优Canny低阈值
        'canny_high_threshold': 90,  # 最优Canny高阈值
        'morphology_kernel_size': 5,  # 最优形态学操作核大小
        'dilate_iterations': 3  # 最优膨胀迭代次数
    }

    optimizer = EdgeDetectionOptimizer(img_path1, img_path2, best_params=best_params)
    optimizer.run()
