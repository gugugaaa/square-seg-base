import os, sys, subprocess
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# 1) 数据注册同前
DATA_ROOT = "/kaggle/input/squares-coco/3and5squares"
TRAIN_JSON = os.path.join(DATA_ROOT, "instances_train.json")
VAL_JSON = os.path.join(DATA_ROOT, "instances_val.json")
IMAGE_ROOT = os.path.join("/kaggle/input/squares-yolo/3and5squares/images")

for name in ["square_train", "square_val"]:
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)

register_coco_instances("square_train", {}, TRAIN_JSON, os.path.join(IMAGE_ROOT, "train"))
register_coco_instances("square_val", {}, VAL_JSON, os.path.join(IMAGE_ROOT, "val"))
MetadataCatalog.get("square_train").thing_classes = ["square"]

# 2) 准备 MaskDINO 代码

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from maskdino import add_maskdino_config

cfg = get_cfg()
add_maskdino_config(cfg)
# 3) 合并 COCO 实例分割配置（R50 50ep 示例）
cfg.merge_from_file("MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml")

# 4) 绑定你的数据与超参
cfg.DATASETS.TRAIN = ("square_train",)
cfg.DATASETS.TEST = ("square_val",)
cfg.DATALOADER.NUM_WORKERS = 2

# 可选：预训练权重（建议使用以加速收敛）
# cfg.MODEL.WEIGHTS = "/path/to/maskdino_pretrained.pth"
cfg.MODEL.WEIGHTS = ""

# 从白色背景中分割黑色实心正方形
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = []
cfg.INPUT.MASK_FORMAT = "polygon"

cfg.OUTPUT_DIR = "output/maskdino_square"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

evaluator = COCOEvaluator("square_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "square_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))