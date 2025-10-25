import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from pycocotools.coco import COCO
from PIL import Image

# 数据集路径配置
DATA_ROOT = "datasets/coco/3squares"
TRAIN_JSON = os.path.join(DATA_ROOT, "instances_train.json")
VAL_JSON = os.path.join(DATA_ROOT, "instances_val.json")
IMAGE_ROOT = os.path.join("datasets/yolo/3squares/images")

# 模型配置
MODEL_NAME = "facebook/mask2former-swin-small-coco-instance"
OUTPUT_DIR = "./mask2former-3squares"
NUM_CLASSES = 1  # 正方形类别数（不包括背景）

class COCOInstanceDataset(Dataset):
    """COCO 格式的实例分割数据集"""
    
    def __init__(self, json_path, image_root, processor, split):
        self.coco = COCO(json_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_root = os.path.join(image_root, split)
        self.processor = processor
        
        # 创建类别映射（将 COCO category_id 映射到 0-based 索引）
        coco_categories = self.coco.getCatIds()
        self.category_mapping = {coco_id: 0 for coco_id in coco_categories}
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # 获取图像信息
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.image_root, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        
        # 获取标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 准备 masks 和 class_labels
        masks = []
        class_labels = []
        for ann in anns:
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            # 使用映射后的类别 ID
            class_labels.append(self.category_mapping[ann['category_id']])
        
        # 使用 processor 处理图像
        inputs = self.processor(images=image, return_tensors="pt")
        
        # 移除批次维度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # 添加标签（修正格式）
        if len(masks) > 0:
            # mask_labels 使用 float32 类型而不是 long
            inputs["mask_labels"] = [torch.tensor(mask, dtype=torch.float32) for mask in masks]
            inputs["class_labels"] = torch.tensor(class_labels, dtype=torch.long)
        else:
            # 处理没有标注的情况
            inputs["mask_labels"] = []
            inputs["class_labels"] = torch.tensor([], dtype=torch.long)
        
        return inputs

def collate_fn(batch):
    """整理到一个批次中"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    
    # 正确处理 mask_labels 和 class_labels
    mask_labels = []
    class_labels = []
    
    for item in batch:
        # 每个样本的 masks 堆叠成 [num_instances, height, width]
        if len(item["mask_labels"]) > 0:
            sample_masks = torch.stack(item["mask_labels"])  # [num_instances, H, W]
            sample_labels = item["class_labels"]  # [num_instances]
        else:
            # 处理空标注的情况：创建一个虚拟的空mask，使用 float32 类型
            sample_masks = torch.zeros((1, pixel_values.shape[-2], pixel_values.shape[-1]), dtype=torch.float32)
            sample_labels = torch.tensor([0], dtype=torch.long)
        
        mask_labels.append(sample_masks)
        class_labels.append(sample_labels)
    
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,  # 保持为列表，每个元素是 [num_instances, H, W]
        "class_labels": class_labels,  # 保持为列表，每个元素是 [num_instances]
    }

def main():
    # 1. 初始化处理器和模型
    print("加载模型和处理器...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    # 2. 创建数据集
    print("加载数据集...")
    train_dataset = COCOInstanceDataset(TRAIN_JSON, IMAGE_ROOT, processor, split="train")
    val_dataset = COCOInstanceDataset(VAL_JSON, IMAGE_ROOT, processor, split="val")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 3. 训练配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        num_train_epochs=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # 只保留2个最佳模型
        logging_steps=10,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,  # loss越小越好
        fp16=True,
    )
    
    # 4. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)], # 早停耐心
    )
    
    # 5. 开始训练
    print("开始训练...")
    trainer.train()
    
    # 6. 保存最终模型（此时已经是最佳模型）
    print("保存最佳模型...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
    print(f"训练完成！最佳模型已保存到: {os.path.join(OUTPUT_DIR, 'best_model')}")

if __name__ == "__main__":
    
    # from transformers import AutoImageProcessor

    # processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
    # dataset = COCOInstanceDataset(VAL_JSON, IMAGE_ROOT, processor, split="val")

    # print(f"数据集大小: {len(dataset)}")
    # print("测试加载第一个样本...")
    # sample = dataset[0]
    # print(f"样本keys: {sample.keys()}")
    # print(f"mask数量: {len(sample['mask_labels'])}")
    # print(f"类别标签: {sample['class_labels']}")

    main()
