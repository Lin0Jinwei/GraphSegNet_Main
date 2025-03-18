import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
from skimage.metrics import structural_similarity as ssim

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 输入和输出文件夹路径
INPUT_DIR = r'test_result/test_20250317_200342/1_result'  # 输入文件夹路径
OUTPUT_DIR = r'./canny_result/1'  # 输出文件夹路径

# 检查目录是否存在，如果不存在则创建它
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 载入贝叶斯优化返回的最佳参数
BEST_PARAMS = {
    'blur_kernel': 3,  # 最优模糊核大小
    'canny_low_threshold': 50,  # 最优Canny低阈值
    'canny_high_threshold': 90,  # 最优Canny高阈值
    'morphology_kernel_size': 5,  # 最优形态学操作核大小
    'dilate_iterations': 3  # 最优膨胀迭代次数
}


def calculate_iou(mask_true, mask_pred):
    """
    计算交并比（IoU）。
    :param mask_true: 真实掩码
    :param mask_pred: 预测掩码
    :return: IoU 值
    """
    mask_true_bool = mask_true.astype(bool)
    mask_pred_bool = mask_pred.astype(bool)
    intersection = np.logical_and(mask_true_bool, mask_pred_bool).sum()
    union = np.logical_or(mask_true_bool, mask_pred_bool).sum()
    if union == 0:
        return 0
    iou = intersection / union
    return iou


def calculate_ssim(img1, img2):
    """
    计算结构相似性（SSIM）。
    :param img1: 图像1
    :param img2: 图像2
    :return: SSIM 值
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 计算SSIM（范围0-1，越大越相似）
    score, _ = ssim(gray1, gray2, full=True)
    return max(score, 0)  # 确保非负


def process_images(original_path, segmented_path, log_file):
    """
    处理每一对原始图像和分割图像。
    :param original_path: 原始图像路径
    :param segmented_path: 分割图像路径
    :param log_file: 日志文件对象
    """
    # 载入原始和分割后的图像
    img_original = cv2.imread(original_path)
    img_segmented = cv2.imread(segmented_path)

    # 对原始图像应用最佳参数
    img_original_blurred = cv2.GaussianBlur(img_original, (BEST_PARAMS['blur_kernel'], BEST_PARAMS['blur_kernel']), 0)
    edges_true = cv2.Canny(img_original_blurred, BEST_PARAMS['canny_low_threshold'], BEST_PARAMS['canny_high_threshold'])
    edges_true_filled = cv2.morphologyEx(edges_true, cv2.MORPH_CLOSE,
                                         np.ones((BEST_PARAMS['morphology_kernel_size'], BEST_PARAMS['morphology_kernel_size']), np.uint8))
    edges_true_filled = cv2.dilate(edges_true_filled,
                                   np.ones((BEST_PARAMS['morphology_kernel_size'], BEST_PARAMS['morphology_kernel_size']), np.uint8),
                                   iterations=BEST_PARAMS['dilate_iterations'])

    # 对分割图像应用相同的最佳参数
    img_segmented_blurred = cv2.GaussianBlur(img_segmented, (BEST_PARAMS['blur_kernel'], BEST_PARAMS['blur_kernel']), 0)
    edges_pred = cv2.Canny(img_segmented_blurred, BEST_PARAMS['canny_low_threshold'], BEST_PARAMS['canny_high_threshold'])
    edges_pred_filled = cv2.morphologyEx(edges_pred, cv2.MORPH_CLOSE,
                                         np.ones((BEST_PARAMS['morphology_kernel_size'], BEST_PARAMS['morphology_kernel_size']), np.uint8))
    edges_pred_filled = cv2.dilate(edges_pred_filled,
                                   np.ones((BEST_PARAMS['morphology_kernel_size'], BEST_PARAMS['morphology_kernel_size']), np.uint8),
                                   iterations=BEST_PARAMS['dilate_iterations'])

    # 计算IoU
    final_iou = calculate_iou(edges_true_filled, edges_pred_filled)
    log_file.write(f"Final IoU for {os.path.basename(original_path)}: {final_iou:.4f}\n")
    print(f"Final IoU after optimization: {final_iou:.4f}")

    # 创建交集叠加图片
    intersection = np.logical_and(edges_true_filled, edges_pred_filled).astype(np.uint8) * 255
    intersection_iou = calculate_iou(edges_true_filled, intersection)
    log_file.write(f"Intersection IoU for {os.path.basename(original_path)}: {intersection_iou:.4f}\n")
    print(f"Intersection IoU after optimization: {intersection_iou:.4f}")

    # 计算SSIM
    ssim_score = calculate_ssim(img_original, img_segmented)
    log_file.write(f"SSIM for {os.path.basename(original_path)}: {ssim_score:.3f}\n")

    # 保存结果
    final_iou_list.append(final_iou)
    intersection_iou_list.append(intersection_iou)
    ssim_list.append(ssim_score)

    # 保存图像结果
    file_name = os.path.basename(original_path).split('.')[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{file_name}_true_best.png'), edges_true_filled)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{file_name}_pred_best.png'), edges_pred_filled)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{file_name}_intersection.png'), intersection)
    print(f"Processed and saved results for {file_name}.")

    # 可视化
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title("原始图像")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB))
    plt.title("预测图像")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(intersection, cmap='gray')
    plt.title(f"交集 (IoU: {final_iou:.4f}, 交集 IoU: {intersection_iou:.4f})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{file_name}_comparison.png"))
    plt.close()


# 初始化指标列表
final_iou_list = []
intersection_iou_list = []
ssim_list = []

# 获取输入文件夹内所有图片对
original_images = sorted(glob.glob(os.path.join(INPUT_DIR, '*_original.png')))
segmented_images = sorted(glob.glob(os.path.join(INPUT_DIR, '*_predicted.png')))

# 创建日志文件记录每对图像的指标
with open(os.path.join(OUTPUT_DIR, 'iou_log.txt'), 'w') as log_file:
    log_file.write("IoU 计算日志\n")
    log_file.write("=================================\n")

    # 处理每对原图和预测图
    for original_path in original_images:
        prefix = original_path.replace('_original.png', '')
        corresponding_predicted_image = prefix + '_predicted.png'
        if corresponding_predicted_image in segmented_images:
            process_images(original_path, corresponding_predicted_image, log_file)

    # 计算平均指标
    average_final_iou = np.mean(final_iou_list) if final_iou_list else 0.0
    average_intersection_iou = np.mean(intersection_iou_list) if intersection_iou_list else 0.0
    average_ssim = np.mean(ssim_list) if ssim_list else 0.0

    log_file.write(f"\n平均 Final IoU: {average_final_iou:.4f}\n")
    log_file.write(f"平均 Intersection IoU: {average_intersection_iou:.4f}\n")
    log_file.write(f"平均 SSIM: {average_ssim:.4f}\n")

print("批处理完成。")

# 打印平均指标
print(f"平均 Final IoU: {average_final_iou:.4f}")
print(f"平均 Intersection IoU: {average_intersection_iou:.4f}")
print(f"平均 SSIM: {average_ssim:.4f}")

# 可视化指标
plt.figure(figsize=(15, 10))
plt.subplot(2, 4, 7)
plt.barh(['Final IoU', 'Intersection IoU', 'SSIM'],
         [average_final_iou, average_intersection_iou, average_ssim],
         color=['#1f77b4', '#ff7f0e', '#2ca02c'])

plt.title("评估指标概览")
plt.xlabel("得分")
plt.xlim(0, 1.2)

# 添加数值标签
for i, v in enumerate([average_final_iou, average_intersection_iou, average_ssim]):
    plt.text(v + 0.02, i, f"{v:.2f}", va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_metrics.png"))
plt.close()