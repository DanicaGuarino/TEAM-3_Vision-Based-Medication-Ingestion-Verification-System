from ultralytics import YOLO

# load pretrained YOLO model
model = YOLO("yolov8n.pt")

# train on pill dataset
model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
