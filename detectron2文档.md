# Detectron2 实例分割完整指南

## 注册coco格式数据集

```python
from detectron2.data.datasets import register_coco_instances

# 注册 COCO 实例分割数据集
register_coco_instances(
    "square_train",       # 数据集名
    {},                            # 可选元数据字典
    "path/to/annotations.json",    # COCO JSON 标注文件
    "path/to/image_directory"      # 图像目录
)
```

### coco数据集，多边形标注

“segmentation” 字段是一个由多边形顶点数组组成的列表，每个顶点数组以 [x1, y1, x2, y2, …, xn, yn] 的形式排列，用于描绘对象的轮廓.​

如果一个实例由多个不连续区域或带孔多边形表示，可在 “segmentation” 中包含多个顶点数组.

## 配置与训练

```python
import numpy as np
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

# 加载预训练配置
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)

# 数据集配置
cfg.DATASETS.TRAIN = ("square_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

# 模型和权重配置
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有一个类别

# 训练参数
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000

# 输入格式
cfg.INPUT.MASK_FORMAT = "polygon"

# 启动训练
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

### 关键配置说明

- 使用预训练 Mask R-CNN 配置文件作为基础
- 设置训练集为 `"square_train"`
- 指定类别数为 1（仅正方形）
- 采用多边形分割格式
- 设置合理的学习率和批大小以适应小数据集