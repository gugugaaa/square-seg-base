import os
import json
import glob
import cv2
import numpy as np

# 数据集根目录
DATASET_DIR = '../datasets/yolo/3squares'
# 输出 COCO 文件目录
OUTPUT_DIR = '../datasets/coco/3squares'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COCO categories
CATEGORIES = [{
    "id": 1,
    "name": "square",
    "supercategory": "shape"
}]

def polygon_area(coords):
    # Shoelace formula
    x = coords[0::2]
    y = coords[1::2]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2

def convert_split(split):
    images, annotations = [], []
    ann_id = 1
    img_files = sorted(glob.glob(f'{DATASET_DIR}/images/{split}/*.jpg'))
    for img_id, img_path in enumerate(img_files, 1):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        images.append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": filename
        })
        label_path = f'{DATASET_DIR}/labels/{split}/{os.path.splitext(filename)[0]}.txt'
        with open(label_path) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                # parts: [class, x1, y1, ..., x4, y4]
                coords_norm = parts[1:]
                coords = [coords_norm[i] * (w if i%2==0 else h) for i in range(len(coords_norm))]
                area = polygon_area(coords)
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, y_min = min(xs), min(ys)
                bbox = [x_min, y_min, max(xs)-x_min, max(ys)-y_min]
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(parts[0]) + 1,  # 类别ID从1开始
                    "segmentation": [coords],
                    "area": float(area),
                    "bbox": [float(x) for x in bbox],
                    "iscrowd": 0
                })
                ann_id += 1
    coco_dict = {
        "info": {"description": "3squares dataset for coco"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }
    with open(os.path.join(OUTPUT_DIR, f'instances_{split}.json'), 'w') as jf:
        json.dump(coco_dict, jf, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        convert_split(split)
    print('COCO 格式转换完成！')
