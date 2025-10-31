import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def group_annotations_by_image(annotations):
    image_groups = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_groups:
            image_groups[img_id] = []
        image_groups[img_id].append(ann)
    return image_groups

def load_image(image_path_dir, image_id):
    # 假设图片名为"{image_id}.jpg"
    img_filename = os.path.join(image_path_dir, f"img_{image_id}.jpg")
    if not os.path.exists(img_filename):
        print(f"未找到图片 {img_filename}")
        return None
    return Image.open(img_filename).convert('RGB')

def visualize_image_with_masks(image, annotations, fig_title=""):
    if image is None or not annotations:
        return
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    
    # 叠加所有mask，半透明红色
    for ann in annotations:
        rle = ann['segmentation']
        mask = maskUtils.decode(rle).astype(np.uint8) * 255
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_rgb[:, :, 0] = mask  # 红色通道
        
        # 如有需要，调整mask到图片大小（假设一致）
        mask_resized = Image.fromarray(mask_rgb).resize(image.size, Image.Resampling.NEAREST)
        mask_overlay = np.array(mask_resized)
        
        ax.imshow(mask_overlay, alpha=0.5)
        
        # 可选：画bbox
        bbox = ann['bbox']
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, f"Score {ann['score']:.2f}", color='blue', fontsize=10)
    
    ax.set_title(fig_title)
    ax.axis('off')
    return fig

def main():
    parser = argparse.ArgumentParser(description="可视化MaskDINO预测结果")
    parser.add_argument('json_path', type=str, help='标注JSON路径')
    parser.add_argument('image_path', type=str, help='图片文件夹路径')
    parser.add_argument('--full_visualize', action='store_true', default=False,
                        help='如设定则全部可视化并保存，否则只显示前4张')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='保存可视化结果的文件夹')
    
    args = parser.parse_args()
    
    annotations = load_annotations(args.json_path)
    image_groups = group_annotations_by_image(annotations)
    unique_image_ids = sorted(list(image_groups.keys()))
    
    if not args.full_visualize:
        # 默认：显示前4张图片
        batch_size = 4
        num_to_show = min(batch_size, len(unique_image_ids))
        figs = []
        for i in range(num_to_show):
            img_id = unique_image_ids[i]
            image = load_image(args.image_path, img_id)
            title = f"img_{img_id}"
            fig = visualize_image_with_masks(image, image_groups[img_id], title)
            figs.append(fig)
        
        plt.show()
    else:
        # 全部可视化并保存
        os.makedirs(args.output_dir, exist_ok=True)
        for img_id in unique_image_ids:
            image = load_image(args.image_path, img_id)
            if image is None:
                continue
            title = f"img{img_id}"
            fig = visualize_image_with_masks(image, image_groups[img_id], title)
            output_path = os.path.join(args.output_dir, f"vis_{img_id}.png")
            fig.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
        print(f"全部可视化结果已保存到 {args.output_dir}")

if __name__ == "__main__":
    main()
