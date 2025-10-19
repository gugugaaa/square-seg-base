from ultralytics import YOLO
model = YOLO("yolo12n-obb.yaml")  # 预训练
results = model.train(data="data.yaml", epochs=100, imgsz=320, batch=16)