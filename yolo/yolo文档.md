# YOLO12 Seg 极简摘要

基于最新文档（YOLO12 支持 seg 任务，**未开放预训练权重**，需从 [Ultralytics GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/12/) 下载 YAML 文件自行训练；API 与 YOLO11 一致，无重大差异）。  
**焦点：数据集格式和 API，针对自定义旋转重叠正方形任务。**

注：测试发现OBB不适合该分割任务
---

## 数据集格式（通用）

**结构：**
- `dataset/`
  - `images/train/`
  - `images/val/`
  - `labels/train/`
  - `labels/val/`
- 图像为 JPG/PNG
- 标签为 TXT（每图像一个 TXT 文件，同名，如 `img_0.jpg` → `img_0.txt`）

**YAML 配置（data.yaml）：**
```yaml
yamlpath: /path/to/dataset  # 根目录
train: images/train         # 训练图像路径
val: images/val             # 验证图像路径
nc: 1                       # 类数（1 为 square）
names: ['square']           # 类名
```

---

## Seg 标签格式（实例分割）

- 每行一个对象：`class_id` + 多边形点（归一化 0-1，按图像宽高）
- 对于旋转正方形：`class 0 x1 y1 x2 y2 x3 y3 x4 y4`（顺/逆时针顺序，4 点够用；支持更多点捕获复杂形状）

**示例 TXT：**
```
0 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6
```

---

**注意：**
- 标签点需处理重叠（每个对象独立标注）
- 合成数据可自动化计算点
- 转换工具：`JSON2YOLO`（COCO 转）

---

## API（训练/验证，Python）

**安装：**
```bash
pip install "ultralytics" "numpy<2.0.0"
```
老版本matplotlib不支持numpy2+

**Seg 训练：**
```python
from ultralytics import YOLO
model = YOLO("yolo12-seg.yaml")  # 下载 yaml 文件，自行训练
results = model.train(data="data.yaml", epochs=100, imgsz=640, batch=16)
```

**验证（评估 mAP、IoU 等）：**
```python
model = YOLO("runs/segment/train/weights/best.pt")
metrics = model.val(data="data.yaml", split='test')  # 访问 metrics.seg.map
# 默认split='val'
```

**其他：**
- 预测：`model.predict(source="img.jpg")`
- 结果访问：`result.masks`
- 数据增强自动支持旋转/重叠

---