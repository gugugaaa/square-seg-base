import os
import json
import shutil
from pathlib import Path

def merge_yolo_datasets(dataset1_path, dataset2_path, output_path):
    """合并两个 YOLO 格式数据集"""
    print(f"开始合并 YOLO 数据集...")
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{output_path}/images/{split}', exist_ok=True)
        os.makedirs(f'{output_path}/labels/{split}', exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        print(f"处理 {split} split...")
        
        # 复制第一个数据集的文件
        dataset1_img_dir = f'{dataset1_path}/images/{split}'
        dataset1_label_dir = f'{dataset1_path}/labels/{split}'
        
        counter = 0
        if os.path.exists(dataset1_img_dir):
            for img_file in sorted(os.listdir(dataset1_img_dir)):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    # 复制图片
                    src_img = os.path.join(dataset1_img_dir, img_file)
                    dst_img = f'{output_path}/images/{split}/img_{counter}.jpg'
                    shutil.copy2(src_img, dst_img)
                    
                    # 复制标签
                    base_name = os.path.splitext(img_file)[0]
                    src_label = os.path.join(dataset1_label_dir, f'{base_name}.txt')
                    if os.path.exists(src_label):
                        dst_label = f'{output_path}/labels/{split}/img_{counter}.txt'
                        shutil.copy2(src_label, dst_label)
                    
                    counter += 1
        
        # 复制第二个数据集的文件（从 counter 开始编号）
        dataset2_img_dir = f'{dataset2_path}/images/{split}'
        dataset2_label_dir = f'{dataset2_path}/labels/{split}'
        
        if os.path.exists(dataset2_img_dir):
            for img_file in sorted(os.listdir(dataset2_img_dir)):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    # 复制图片
                    src_img = os.path.join(dataset2_img_dir, img_file)
                    dst_img = f'{output_path}/images/{split}/img_{counter}.jpg'
                    shutil.copy2(src_img, dst_img)
                    
                    # 复制标签
                    base_name = os.path.splitext(img_file)[0]
                    src_label = os.path.join(dataset2_label_dir, f'{base_name}.txt')
                    if os.path.exists(src_label):
                        dst_label = f'{output_path}/labels/{split}/img_{counter}.txt'
                        shutil.copy2(src_label, dst_label)
                    
                    counter += 1
        
        print(f"  {split}: 合并了 {counter} 张图片")
    
    # 生成 data.yaml
    yaml_content = f"""path: {output_path}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['square']
"""
    with open(f'{output_path}/data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("YOLO 数据集合并完成！")

def merge_coco_datasets(dataset1_path, dataset2_path, output_path):
    """合并两个 COCO 格式数据集"""
    print(f"开始合并 COCO 数据集...")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        print(f"处理 {split} split...")
        
        # 读取两个数据集的 JSON 文件
        json1_path = f'{dataset1_path}/instances_{split}.json'
        json2_path = f'{dataset2_path}/instances_{split}.json'
        
        if not os.path.exists(json1_path) or not os.path.exists(json2_path):
            print(f"  跳过 {split}: JSON 文件不存在")
            continue
        
        with open(json1_path, 'r') as f:
            coco1 = json.load(f)
        with open(json2_path, 'r') as f:
            coco2 = json.load(f)
        
        # 创建合并后的 COCO 字典
        merged_coco = {
            "info": {"description": "Merged COCO dataset"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": coco1.get("categories", [])
        }
        
        # 复制第一个数据集
        img_counter = 0
        ann_counter = 0
        
        # 处理第一个数据集的图片
        img_id_map1 = {}
        for img in coco1.get("images", []):
            old_id = img["id"]
            new_id = img_counter + 1
            img_id_map1[old_id] = new_id
            
            # 复制图片文件
            src_img = os.path.join(dataset1_path, img["file_name"])
            if os.path.exists(src_img):
                dst_img = os.path.join(output_path, f'img_{img_counter}.jpg')
                shutil.copy2(src_img, dst_img)
                
                # 添加图片信息
                merged_coco["images"].append({
                    "id": new_id,
                    "width": img["width"],
                    "height": img["height"],
                    "file_name": f'img_{img_counter}.jpg'
                })
                img_counter += 1
        
        # 处理第一个数据集的标注
        for ann in coco1.get("annotations", []):
            if ann["image_id"] in img_id_map1:
                ann_counter += 1
                merged_coco["annotations"].append({
                    "id": ann_counter,
                    "image_id": img_id_map1[ann["image_id"]],
                    "category_id": ann["category_id"],
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "bbox": ann["bbox"],
                    "iscrowd": ann.get("iscrowd", 0)
                })
        
        # 处理第二个数据集的图片（从 img_counter 开始编号）
        img_id_map2 = {}
        for img in coco2.get("images", []):
            old_id = img["id"]
            new_id = img_counter + 1
            img_id_map2[old_id] = new_id
            
            # 复制图片文件
            src_img = os.path.join(dataset2_path, img["file_name"])
            if os.path.exists(src_img):
                dst_img = os.path.join(output_path, f'img_{img_counter}.jpg')
                shutil.copy2(src_img, dst_img)
                
                # 添加图片信息
                merged_coco["images"].append({
                    "id": new_id,
                    "width": img["width"],
                    "height": img["height"],
                    "file_name": f'img_{img_counter}.jpg'
                })
                img_counter += 1
        
        # 处理第二个数据集的标注
        for ann in coco2.get("annotations", []):
            if ann["image_id"] in img_id_map2:
                ann_counter += 1
                merged_coco["annotations"].append({
                    "id": ann_counter,
                    "image_id": img_id_map2[ann["image_id"]],
                    "category_id": ann["category_id"],
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "bbox": ann["bbox"],
                    "iscrowd": ann.get("iscrowd", 0)
                })
        
        # 保存合并后的 JSON
        output_json = os.path.join(output_path, f'instances_{split}.json')
        with open(output_json, 'w') as f:
            json.dump(merged_coco, f, ensure_ascii=False, indent=2)
        
        print(f"  {split}: 合并了 {img_counter} 张图片, {ann_counter} 个标注")
    
    print("COCO 数据集合并完成！")

def main(format_type, dataset1, dataset2, output):
    """主函数"""
    if format_type.lower() == 'yolo':
        merge_yolo_datasets(dataset1, dataset2, output)
    elif format_type.lower() == 'coco':
        merge_coco_datasets(dataset1, dataset2, output)
    else:
        print(f"不支持的格式: {format_type}，请使用 'yolo' 或 'coco'")

if __name__ == '__main__':
    
    FORMAT = 'yolo'

    DATASET1 = 'datasets/yolo/3squares'
    DATASET2 = 'datasets/yolo/5squares'
    OUTPUT = 'datasets/yolo/3and5squares'

    main(FORMAT, DATASET1, DATASET2, OUTPUT)
