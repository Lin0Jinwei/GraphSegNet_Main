from skopt import gp_minimize
from skopt.space import Integer, Categorical
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_images(original_path, segmented_path):
    """加载原始图像和分割图像"""
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(segmented_path)
    return img1, img2


def calculate_iou(mask_true, mask_pred):
    """计算IoU值"""
    mask_true_bool = mask_true.astype(bool)
    mask_pred_bool = mask_pred.astype(bool)
    intersection = np.logical_and(mask_true_bool, mask_pred_bool).sum()
    union = np.logical_or(mask_true_bool, mask_pred_bool).sum()
    if union == 0:
        return 0
    iou = intersection / union
    return iou


def objective(params, img1, img2, iou_list, params_list):
    """定义目标函数（负IoU）"""
    # 模糊操作
    blurred_img1 = cv2.GaussianBlur(img1, (params[0], params[0]), 0)
    edges_true = cv2.Canny(blurred_img1, params[1], params[2])
    filled_edges_true = cv2.morphologyEx(edges_true, cv2.MORPH_CLOSE, np.ones((params[3], params[3]), np.uint8))
    filled_edges_true = cv2.dilate(filled_edges_true, np.ones((params[3], params[3]), np.uint8), iterations=params[4])

    # 对第二张图像进行相同的处理以获得 filled_edges_pred
    blurred_img2 = cv2.GaussianBlur(img2, (params[0], params[0]), 0)
    edges_pred = cv2.Canny(blurred_img2, params[1], params[2])
    filled_edges_pred = cv2.morphologyEx(edges_pred, cv2.MORPH_CLOSE, np.ones((params[3], params[3]), np.uint8))
    filled_edges_pred = cv2.dilate(filled_edges_pred, np.ones((params[3], params[3]), np.uint8), iterations=params[4])

    iou = calculate_iou(filled_edges_true, filled_edges_pred)
    iou_list.append(iou)  # Store the IoU value
    params_list.append(params)  # Store the current parameters

    return -iou  # 最小化负IoU等价于最大化IoU



def optimize_parameters(img1, img2, search_space, n_calls=30):
    """进行贝叶斯优化"""
    iou_list = []  # List to store the IoU values
    params_list = []  # List to store the parameters corresponding to each IoU value
    res_gp = gp_minimize(lambda params: objective(params, img1, img2, iou_list, params_list), search_space, n_calls=n_calls, random_state=0)
    return res_gp, iou_list, params_list



def apply_best_params(img1, img2, best_params):
    """应用最佳参数生成最终的图像结果"""
    blurred_img1_best = cv2.GaussianBlur(img1, (best_params['blur_kernel'], best_params['blur_kernel']), 0)
    edges_true_best = cv2.Canny(blurred_img1_best, best_params['canny_low_threshold'],
                                best_params['canny_high_threshold'])
    filled_edges_true_best = cv2.morphologyEx(edges_true_best, cv2.MORPH_CLOSE, np.ones(
        (best_params['morphology_kernel_size'], best_params['morphology_kernel_size']), np.uint8))
    filled_edges_true_best = cv2.dilate(filled_edges_true_best, np.ones(
        (best_params['morphology_kernel_size'], best_params['morphology_kernel_size']), np.uint8),
                                        iterations=best_params['dilate_iterations'])

    blurred_img2_best = cv2.GaussianBlur(img2, (best_params['blur_kernel'], best_params['blur_kernel']), 0)
    edges_pred_best = cv2.Canny(blurred_img2_best, best_params['canny_low_threshold'],
                                best_params['canny_high_threshold'])
    filled_edges_pred_best = cv2.morphologyEx(edges_pred_best, cv2.MORPH_CLOSE, np.ones(
        (best_params['morphology_kernel_size'], best_params['morphology_kernel_size']), np.uint8))
    filled_edges_pred_best = cv2.dilate(filled_edges_pred_best, np.ones(
        (best_params['morphology_kernel_size'], best_params['morphology_kernel_size']), np.uint8),
                                        iterations=best_params['dilate_iterations'])

    return filled_edges_true_best, filled_edges_pred_best


def save_images(output_dir, filled_edges_true_best, filled_edges_pred_best, intersection, iou_values, params_list):
    """保存最终结果和交集图片，同时保存每个IoU及对应的参数"""
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, '1_true_best.png'), filled_edges_true_best)
    cv2.imwrite(os.path.join(output_dir, '1_pred_best.png'), filled_edges_pred_best)
    cv2.imwrite(os.path.join(output_dir, 'intersection.png'), intersection)

    # 计算IoU均值
    iou_mean = np.mean(iou_values)

    # 保存IoU值和对应的参数到文本文件
    with open(os.path.join(output_dir, 'iou_values_and_params.txt'), 'w') as f:
        for iou, params in zip(iou_values, params_list):
            f.write(f"IoU: {iou:.4f} | Parameters: blur_kernel={params[0]}, canny_low_threshold={params[1]}, "
                    f"canny_high_threshold={params[2]}, morphology_kernel_size={params[3]}, "
                    f"dilate_iterations={params[4]}\n")

        # 最后写入IoU均值
        f.write(f"IoU Mean: {iou_mean:.4f}\n")

    print(f"IoU values and parameters saved to {os.path.join(output_dir, 'iou_values_and_params.txt')}")




def visualize_results(filled_edges_true_best, filled_edges_pred_best, intersection):
    """可视化最终结果和交集叠加图片"""
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    axs[0].imshow(filled_edges_true_best, cmap='gray')
    axs[0].set_title('Filled True Edges (Optimized)')
    axs[0].axis('off')

    axs[1].imshow(filled_edges_pred_best, cmap='gray')
    axs[1].set_title('Filled Predicted Edges (Optimized)')
    axs[1].axis('off')

    axs[2].imshow(intersection, cmap='gray')
    axs[2].set_title('Intersection of True and Predicted Edges')
    axs[2].axis('off')

    plt.show()


def optimize_and_visualize(original_path, segmented_path, search_space, n_calls=20, output_dir='output_images'):
    """主函数：加载图像，进行优化，生成结果并可视化"""
    img1, img2 = load_images(original_path, segmented_path)

    # 进行贝叶斯优化
    res_gp, iou_list, params_list = optimize_parameters(img1, img2, search_space, n_calls)

    # 输出最佳参数和IoU
    print("Best parameters found:", res_gp.x)
    print("Best IoU:", -res_gp.fun)

    # 使用最佳参数生成最终结果
    best_params = {
        'blur_kernel': res_gp.x[0],
        'canny_low_threshold': res_gp.x[1],
        'canny_high_threshold': res_gp.x[2],
        'morphology_kernel_size': res_gp.x[3],
        'dilate_iterations': res_gp.x[4]
    }

    # 应用最佳参数生成图像
    filled_edges_true_best, filled_edges_pred_best = apply_best_params(img1, img2, best_params)

    # 计算最终IoU
    final_iou = calculate_iou(filled_edges_true_best, filled_edges_pred_best)
    print(f"Final IoU after optimization: {final_iou:.4f}")

    # 创建交集叠加图片
    intersection = np.logical_and(filled_edges_true_best, filled_edges_pred_best).astype(np.uint8) * 255
    intersection_iou = calculate_iou(filled_edges_true_best, intersection)
    print(f"Intersection IoU after optimization: {intersection_iou:.4f}")

    # 可视化结果
    visualize_results(filled_edges_true_best, filled_edges_pred_best, intersection)

    # 获取子文件夹名称（基于segmented_image_path的父文件夹名称）
    parent_folder_name = os.path.basename(os.path.dirname(segmented_path))

    # 创建输出目录并保存结果和IoU值
    final_output_dir = os.path.join(output_dir, parent_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)

    save_images(final_output_dir, filled_edges_true_best, filled_edges_pred_best, intersection, iou_list, params_list)
    print(f"Final results and intersection saved to: {final_output_dir}")



# 运行主函数
if __name__ == "__main__":
    original_image_path = r'test_result/evaluation_20250317_185623/1_result/cropped_resized_10_original_original.png'
    segmented_image_path = r'test_result/evaluation_20250317_185623/1_result/cropped_resized_10_original_predicted.png'
    search_space = [
        Categorical([3, 5, 7], name='blur_kernel'),
        Integer(10, 90, name='canny_low_threshold'),
        Integer(90, 200, name='canny_high_threshold'),
        Integer(3, 5, name='morphology_kernel_size'),
        Integer(1, 3, name='dilate_iterations')
    ]

    optimize_and_visualize(original_image_path, segmented_image_path, search_space)


