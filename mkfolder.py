import os
import shutil
import random
#将得到的图像划分为训练集与验证集

# 定义文件夹路径和划分比例
combined_dir = r'./combined'
# uncombined_dir = r'./uncombined'
train_shutter_dir = './train/shutter'
# train_common_dir = './train/common'
test_shutter_dir = './test/shutter'
# test_common_dir = './test/common'

split_ratio = 0.8

image_files = [os.path.join(combined_dir, f) for f in os.listdir(combined_dir) if f.endswith('.png')]

# 随机打乱
random.shuffle(image_files)

train_size = int(split_ratio * len(image_files))
test_size = len(image_files) - train_size

# 移动图像到训练集文件夹
for img_file in image_files[:train_size]:
    shutil.move(img_file, os.path.join(train_shutter_dir, os.path.basename(img_file)))

# 移动图像到验证集文件夹
for img_file in image_files[train_size:]:
    shutil.move(img_file, os.path.join(test_shutter_dir, os.path.basename(img_file)))

print("训练集样本数:", train_size)
print("验证集样本数:", test_size)