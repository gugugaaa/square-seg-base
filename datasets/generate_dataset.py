import cv2
import numpy as np
import random
import os
from shapely.geometry import Polygon

# 数据集根目录变量
DATASET_DIR = 'datasets/yolo/3squares'

def calculate_iou(box1, box2):
    """计算两个正方形的相对 IOU，检测包含或重叠关系"""
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    intersection = poly1.intersection(poly2).area
    
    # 使用相对面积比：intersection相对于较小的多边形的比例
    area1 = poly1.area
    area2 = poly2.area
    min_area = min(area1, area2)
    
    # 返回交集占较小多边形的比例，这样可以检测包含关系
    return intersection / min_area if min_area > 0 else 0

def generate_image_and_labels(num_squares=3, img_size=640, is_train=True):
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # 白底
    labels = []  # 存储标签行
    boxes = []  # 存储已生成的正方形框
    
    for _ in range(num_squares):
        for _ in range(100):
            size = random.randint(50, 200)  # 随机大小
            center = (random.randint(size//2, img_size - size//2), random.randint(size//2, img_size - size//2))  # 避免越界
            angle = random.uniform(0, 360)  # 随机旋转
            # 创建旋转矩形
            rect = (center, (size, size), angle)
            box = cv2.boxPoints(rect)  # 获取 4 角点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            box = box.astype(np.int32)  # 整数化
            
            # 检查边界
            if not np.all(
                (box[:, 0] >= 0)
                & (box[:, 0] < img_size)
                & (box[:, 1] >= 0)
                & (box[:, 1] < img_size)
            ):
                continue
            
            # 检查 IOU：与所有已有正方形的相对 IOU 都小于
            iou_valid = True
            for existing_box in boxes:
                iou = calculate_iou(box, existing_box)
                if iou >= 0.8:
                    iou_valid = False
                    break
            
            if iou_valid:
                break
        else:
            continue
        
        cv2.fillPoly(img, [box], (0, 0, 0))  # 填充黑色
        boxes.append(box)  # 添加到已生成列表
        
        # 生成标签：class 0 + 归一化角点 (x/img_size, y/img_size)，扁平化
        norm_box = box.astype(np.float32) / img_size  # 归一化
        label_line = [0] + norm_box.flatten().tolist()  # [0, x1, y1, x2, y2, x3, y3, x4, y4]
        labels.append(label_line)
    
    return img, labels

# 创建目录结构
os.makedirs(f'{DATASET_DIR}/images/train', exist_ok=True)
os.makedirs(f'{DATASET_DIR}/images/val', exist_ok=True)
os.makedirs(f'{DATASET_DIR}/images/test', exist_ok=True)
os.makedirs(f'{DATASET_DIR}/labels/train', exist_ok=True)
os.makedirs(f'{DATASET_DIR}/labels/val', exist_ok=True)
os.makedirs(f'{DATASET_DIR}/labels/test', exist_ok=True)

# 生成训练集 (800 张)
generated_count = 0
attempt = 0
while generated_count < 800:
    img, labels = generate_image_and_labels()
    if len(labels) > 0:  # 确保至少生成了一个正方形
        cv2.imwrite(f'{DATASET_DIR}/images/train/img_{generated_count}.jpg', img)
        with open(f'{DATASET_DIR}/labels/train/img_{generated_count}.txt', 'w') as f:
            for label in labels:
                f.write(' '.join(f'{x:.6f}' for x in label) + '\n')
        generated_count += 1
    attempt += 1
    if attempt > 10000:  # 防止无限循环
        print(f"警告: 只生成了 {generated_count} 张训练图片")
        break

# 生成验证集 (200 张)
generated_count = 0
attempt = 0
while generated_count < 200:
    img, labels = generate_image_and_labels(is_train=False)
    if len(labels) > 0:
        cv2.imwrite(f'{DATASET_DIR}/images/val/img_{generated_count}.jpg', img)
        with open(f'{DATASET_DIR}/labels/val/img_{generated_count}.txt', 'w') as f:
            for label in labels:
                f.write(' '.join(f'{x:.6f}' for x in label) + '\n')
        generated_count += 1
    attempt += 1
    if attempt > 10000:
        print(f"警告: 只生成了 {generated_count} 张验证图片")
        break

# 生成测试集 (100 张)
generated_count = 0
attempt = 0
while generated_count < 100:
    img, labels = generate_image_and_labels(is_train=False)
    if len(labels) > 0:
        cv2.imwrite(f'{DATASET_DIR}/images/test/img_{generated_count}.jpg', img)
        with open(f'{DATASET_DIR}/labels/test/img_{generated_count}.txt', 'w') as f:
            for label in labels:
                f.write(' '.join(f'{x:.6f}' for x in label) + '\n')
        generated_count += 1
    attempt += 1
    if attempt > 10000:
        print(f"警告: 只生成了 {generated_count} 张测试图片")
        break

# 生成 data.yaml
yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['square']
"""
with open('data.yaml', 'w') as f:
    f.write(yaml_content)

print("数据集生成完成！")