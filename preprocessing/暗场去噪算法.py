import cv2
from datetime import datetime
import os

class MaskImageProcessor:
    """
    处理图像的类，用于根据暗场图像（Darkfield）生成掩码并处理原始图像与背景图像的融合。
    """

    def __init__(self, original_img_path, darkfield_img_path, background_img_path, mask_threshold):
        """
        初始化方法，加载原始图像、暗场图像和背景图像，并设置阈值。

        :param original_img_path: 原始图像路径
        :param darkfield_img_path: 暗场图像路径
        :param background_img_path: 背景图像路径
        :param mask_threshold: 用于生成掩码的阈值
        """
        self.original_img = cv2.imread(original_img_path)  # 加载原始图像
        self.darkfield_img = cv2.imread(darkfield_img_path)  # 加载暗场图像
        self.background_img = cv2.imread(background_img_path)  # 加载背景图像
        self.mask_threshold = mask_threshold  # 设置掩码阈值

        # 调整背景图像的大小与原始图像相同
        self.background_img = cv2.resize(self.background_img,
                                         (self.original_img.shape[1], self.original_img.shape[0]))

    def process_images(self):
        """
        处理图像，生成掩码并根据掩码提取原始图像和背景图像的部分，合并为最终结果图像。

        :return: 返回合并后的结果图像、掩码、反转掩码、原始图像部分、背景图像部分、灰度暗场图像
        """
        # 将暗场图像转换为灰度图像，简化后续处理
        gray_darkfield = cv2.cvtColor(self.darkfield_img, cv2.COLOR_BGR2GRAY)

        # 使用Otsu方法自动计算阈值并进行二值化
        _, mask = cv2.threshold(gray_darkfield, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 反转掩码，方便后续处理
        inverse_mask = cv2.bitwise_not(mask)

        # 通过与操作提取原始图像中非亮区域（暗区域）
        result_part = cv2.bitwise_and(self.original_img, self.original_img, mask=inverse_mask)

        # 通过与操作提取背景图像中的亮区域
        background_part = cv2.bitwise_and(self.background_img, self.background_img, mask=mask)

        # 合并结果部分和背景部分
        result = cv2.add(result_part, background_part)

        return result, mask, inverse_mask, result_part, background_part, gray_darkfield

    def save_images(self, result, mask, inverse_mask, result_part, background_part, gray_darkfield):
        """
        保存处理后的图像和中间结果。

        :param result: 合并后的结果图像
        :param mask: 二值掩码
        :param inverse_mask: 反转的掩码
        :param result_part: 原始图像中被掩码遮住的部分
        :param background_part: 背景图像中被掩码遮住的部分
        :param gray_darkfield: 灰度暗场图像
        """
        # 获取当前时间并格式化为文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 设置保存图像的路径
        save_path = './mask_save'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 打印调试信息，方便调试时查看图像尺寸
        print(f'Saving Gray Darkfield shape: {gray_darkfield.shape}')
        print(f'Saving Mask shape: {mask.shape}')
        print(f'Saving Inverse Mask shape: {inverse_mask.shape}')
        print(f'Saving Result Part shape: {result_part.shape}')
        print(f'Saving Background Part shape: {background_part.shape}')
        print(f'Saving Result shape: {result.shape}')

        # 保存各类图像到指定文件夹
        cv2.imwrite(os.path.join(save_path, f'{timestamp}_Gray_Darkfield.jpg'), gray_darkfield)
        cv2.imwrite(os.path.join(save_path, f'{timestamp}_mask.jpg'), mask)
        cv2.imwrite(os.path.join(save_path, f'{timestamp}_inverse_mask.jpg'), inverse_mask)
        cv2.imwrite(os.path.join(save_path, f'{timestamp}_result_part.jpg'), result_part)
        cv2.imwrite(os.path.join(save_path, f'{timestamp}_background_part.jpg'), background_part)
        cv2.imwrite(os.path.join(save_path, f'{timestamp}_result.jpg'), result)


# 使用示例
if __name__ == '__main__':
    # 创建图像处理器对象，传入相关图像路径和阈值
    processor = MaskImageProcessor(original_img_path='transformed_merge_graphene_2020723_0002_C.png',
                                   darkfield_img_path='BF_merge_graphene_2020723_0002_C.jpg',
                                   background_img_path='bg_image_2025-01-06_15-51.jpg',
                                 )

    # 处理图像，获取处理结果
    result, mask, inverse_mask, result_part, background_part, gray_darkfield = processor.process_images()

    # 保存结果图像和中间图像
    processor.save_images(result, mask, inverse_mask, result_part, background_part, gray_darkfield)
