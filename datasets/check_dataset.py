import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 指定目录
image_dir = 'datasets/yolo/5squares/images/test'
label_dir = 'datasets/yolo/5squares/labels/test'

# 获取所有图像文件
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

if len(image_files) < 6:
    print("数据集图像不足6张，无法抽取。")
else:
    # 随机抽取6张
    selected_files = random.sample(image_files, 6)
    print("随机抽取的文件:", ', '.join(selected_files))
    
    # 创建 subplot，2行3列
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, file in enumerate(selected_files):
        # 加载图像
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        height, width = img.shape[:2]
        
        # 加载标签
        label_path = os.path.join(label_dir, file.replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            # 绘制图像
            axs[i].imshow(img)
            
            for label in labels:
                parts = list(map(float, label.strip().split()))
                class_id = int(parts[0])
                points = parts[1:]
                # 重塑为 (n_points, 2)，每个点 (x,y) 反归一化
                points = np.array(points).reshape(-1, 2) * np.array([width, height])
                
                # 绘制多边形
                from matplotlib.patches import Polygon
                poly = Polygon(points, closed=True, edgecolor='r', fill=False, linewidth=2)
                axs[i].add_patch(poly)
            
            axs[i].set_title(f'Image: {file}')
            axs[i].axis('off')
        else:
            axs[i].imshow(img)
            axs[i].set_title(f'Image: {file} (No labels)')
            axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()  # 显示图像
    
    # 可选：保存为文件
    # fig.savefig('dataset_visualization.png')
    # print("可视化已保存到 'dataset_visualization.png'。")