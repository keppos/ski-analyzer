from ultralytics import YOLO


# Load model
model = YOLO("D:/Users/Kevin/VSC Projects/ski-analyzer/runs/detect/train9/weights/last.pt")

# Load image
model.predict("D:/Users/Kevin/VSC Projects/ski-analyzer/data/images/test/snowboarder-test2.png", save=True)