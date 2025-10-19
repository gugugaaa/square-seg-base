from ultralytics import YOLO
model = YOLO("results/ver1-seg-3squares/runs/segment/train/weights/best.pt") 
metrics = model.val(data="data.yaml", split='test')