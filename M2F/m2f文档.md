Mask2Former Transformers 极简文档
一、核心组件
1. Mask2FormerImageProcessor
用于图像预处理和后处理

加载方式：AutoImageProcessor.from_pretrained(model_name)

2. Mask2FormerForUniversalSegmentation
统一的分割模型（支持实例/语义/全景分割）

加载方式：Mask2FormerForUniversalSegmentation.from_pretrained(model_name)

二、推理流程
```python
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torch

# 1. 加载模型和处理器
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-instance")

# 2. 准备图像
image = Image.open("your_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# 3. 推理
with torch.no_grad():
    outputs = model(**inputs)

# 4. 后处理（实例分割）
result = processor.post_process_instance_segmentation(
    outputs, 
    target_sizes=[(image.height, image.width)]
)[0]
# 输出: {"segmentation": tensor, "segments_info": [...]}
```
三、训练流程
```python
from transformers import (
    AutoImageProcessor, 
    Mask2FormerForUniversalSegmentation,
    Trainer, 
    TrainingArguments
)
import torch
from torch.utils.data import Dataset

# 1. 初始化处理器和模型
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-coco-instance",
    num_labels=num_classes,  # 你的类别数
    ignore_mismatched_sizes=True
)

# 2. 数据处理（训练时需要 mask_labels 和 class_labels）
def preprocess_fn(image, masks, class_ids):
    """
    image: PIL Image
    masks: list of binary masks [num_instances, H, W]
    class_ids: list of class IDs for each instance
    """
    inputs = processor(
        images=image,
        return_tensors="pt"
    )
    
    # 准备标签
    inputs["mask_labels"] = [torch.tensor(masks).float()]
    inputs["class_labels"] = [torch.tensor(class_ids).long()]
    
    return inputs

# 3. 训练配置
training_args = TrainingArguments(
    output_dir="./mask2former-finetuned",
    learning_rate=1e-5,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_steps=500,
    logging_steps=100,
    remove_unused_columns=False,
)

# 4. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

trainer.train()
```