import cv2
import numpy as np
import random
import os

# 兼容obb和seg任务
def generate_image_and_labels(num_squares=3, img_size=640, is_train=True):
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # 白底
    labels = []  # 存储标签行
    for _ in range(num_squares):
        for _ in range(100):
            size = random.randint(50, 200)  # 随机大小
            center = (random.randint(size//2, img_size - size//2), random.randint(size//2, img_size - size//2))  # 避免越界
            angle = random.uniform(0, 360)  # 随机旋转
            # 创建旋转矩形
            rect = (center, (size, size), angle)
            box = cv2.boxPoints(rect)  # 获取 4 角点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            box = box.astype(np.int32)  # 整数化
            if np.all(
                (box[:, 0] >= 0)
                & (box[:, 0] < img_size)
                & (box[:, 1] >= 0)
                & (box[:, 1] < img_size)
            ):
                break
        else:
            continue
        cv2.fillPoly(img, [box], (0, 0, 0))  # 填充黑色
        
        # 生成标签：class 0 + 归一化角点 (x/img_size, y/img_size)，扁平化
        norm_box = box.astype(np.float32) / img_size  # 归一化
        label_line = [0] + norm_box.flatten().tolist()  # [0, x1, y1, x2, y2, x3, y3, x4, y4]
        labels.append(label_line)
    
    return img, labels

# 创建目录结构
os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/labels/train', exist_ok=True)
os.makedirs('dataset/labels/val', exist_ok=True)

# 生成训练集 (800 张)
for i in range(800):
    img, labels = generate_image_and_labels()
    cv2.imwrite(f'dataset/images/train/img_{i}.jpg', img)
    with open(f'dataset/labels/train/img_{i}.txt', 'w') as f:
        for label in labels:
            f.write(' '.join(f'{x:.6f}' for x in label) + '\n')

# 生成验证集 (200 张)
for i in range(200):
    img, labels = generate_image_and_labels(is_train=False)
    cv2.imwrite(f'dataset/images/val/img_{i}.jpg', img)
    with open(f'dataset/labels/val/img_{i}.txt', 'w') as f:
        for label in labels:
            f.write(' '.join(f'{x:.6f}' for x in label) + '\n')

# 生成 data.yaml
yaml_content = """
path: dataset
train: images/train
val: images/val
nc: 1
names: ['square']
"""
with open('data.yaml', 'w') as f:
    f.write(yaml_content)

print("数据集生成完成！")