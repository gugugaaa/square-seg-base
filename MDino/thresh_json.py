import json
import os
import argparse

def process_json(input_path, output_path, threshold=0.8):
    """
    处理JSON文件：根据阈值过滤结果
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        threshold: 分数阈值，默认0.5
    """
    # 读取JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据
    filtered_data = []
    image_stats = {}  # 统计每张图片的正方形数量
    
    for item in data:
        # 过滤score小于threshold的项
        if item.get('score', 0) >= threshold:
            
            filtered_data.append(item)
            
            # 统计图片
            image_id = item.get('image_id')
            image_stats[image_id] = image_stats.get(image_id, 0) + 1
    
    # 保存到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    # 打印统计信息
    print("\n=== 图片正方形保留统计 ===")
    # 统计每种保留数量的图片数
    from collections import Counter
    count_stats = Counter(image_stats.values())
    total_images = len(image_stats)
    for count, num_images in sorted(count_stats.items()):
        percent = num_images / total_images * 100
        print(f"保留 {count} 个正方形的图片占比: {percent:.2f}% ({num_images}/{total_images})")
    print(f"\n总计: {len(filtered_data)} 个正方形，来自 {total_images} 张图片")
    
    print(f"处理完成！")

    print(f"已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="过滤并统计正方形检测结果")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出JSON文件路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="分数阈值，默认0.5")
    args = parser.parse_args()

    process_json(args.input, args.output, args.threshold)
