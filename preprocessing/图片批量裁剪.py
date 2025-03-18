import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib

matplotlib.use('TkAgg')

# 用于存储裁剪区域的变量
crop_box = []


def onselect(eclick, erelease):
    global crop_box
    crop_box = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)]


def crop_and_resize_images_in_folder(folder_path, target_size=(1024, 1024)):
    global crop_box

    # 获取文件夹中所有图片文件的路径
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in {folder_path}")
        return

    # 打开第一张图片以获取裁剪区域
    input_image_path1 = os.path.join(folder_path, image_files[0])
    image1 = Image.open(input_image_path1)

    fig, ax = plt.subplots()
    ax.imshow(image1.convert('RGB'))

    # 创建 RectangleSelector
    toggle_selector = RectangleSelector(ax, onselect)

    plt.show()

    if crop_box:
        # 裁剪区域处理
        crop_box = [max(0, x) for x in crop_box]  # 确保裁剪区域不出界

        # 对文件夹中的每一张图片进行裁剪和调整大小
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)

            # 裁剪图片
            cropped_image = image.crop(crop_box)

            # 调整图片为目标尺寸 (1024x1024)
            resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)

            # 保存裁剪并调整尺寸后的图片
            output_image_path = os.path.join(folder_path, f"cropped_resized_{image_file}")
            resized_image.save(output_image_path)

            print(f"Cropped and resized image saved as {output_image_path}")

    else:
        print("No cropping area selected.")


def main():
    folder_path = './datasets/testing_images/500×'  # 替换需要裁剪的图片文件夹路径
    crop_and_resize_images_in_folder(folder_path)


if __name__ == '__main__':
    main()
