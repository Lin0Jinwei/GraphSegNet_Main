import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from bianyuanjiance import EdgeDetectionOptimizer

"""
K-Means 是一种经典的无监督学习算法，用于将数据分为 K 个簇。
它的目标是将数据集中的样本根据特征的相似性分成多个簇，每个簇的中心点尽可能接近簇内样本。
"""

# 1. 读取图像并将其转化为一维数组
im = cv2.imread('input_image.jpeg')
image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 转换为 RGB 色彩空间
pixels = image.reshape((-1, 3))  # 将图像的每个像素点展开为一维

# 2. 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=3, random_state=0)  # 假设要分成 3 类
kmeans.fit(pixels)

# 3. 将每个像素分配到对应的聚类标签
segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)

# 4. 显示分割后的图像
segmented_img = np.uint8(segmented_img)

# 5. 保存分割后的图像
output_folder = 'segmented_output'
os.makedirs(output_folder, exist_ok=True)
segmented_img_path = os.path.join(output_folder, 'segmented_image.jpeg')
cv2.imwrite(segmented_img_path, cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))  # 保存为 BGR 格式

# 6. 保存每个类别的中心颜色（聚类中心）
cluster_centers = kmeans.cluster_centers_
cluster_centers = np.uint8(cluster_centers)

# 打印每个类别的中心颜色，并保存为文本文件
clusters_info_path = os.path.join(output_folder, 'clusters_info.txt')
with open(clusters_info_path, 'w') as f:
    for i, center in enumerate(cluster_centers):
        f.write(f"Cluster {i + 1}: RGB = {center}\n")
output_dir = './segmented_output/train_results'
optimizer = EdgeDetectionOptimizer('cropped_resized_46 (1).jpeg', 'segmented_output/segmented_image.jpeg', output_dir)
optimizer.run()
# 7. 显示分割后的图像
cv2.imshow('Segmented Image', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 输出保存路径和类别信息路径
print(f"Segmented image saved to: {segmented_img_path}")
print(f"Cluster centers saved to: {clusters_info_path}")
