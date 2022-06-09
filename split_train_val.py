# 从训练集数据中分出一部分作为验证集

import os
import random
import shutil

train_path = 'dogvscat/train'
val_path = 'dogvscat/val'
val_ratio = 0.2

if not os.path.exists(val_path):
    os.makedirs(val_path, exist_ok=True)
    imgs_names = os.listdir(train_path)
    random.shuffle(imgs_names)
    num_val = int(0.2 * len(imgs_names))
    val_imgs_names = imgs_names[:num_val]
    for val_image_name in val_imgs_names:
        source = os.path.join(train_path, val_image_name)
        target = val_path
        shutil.move(source, target)