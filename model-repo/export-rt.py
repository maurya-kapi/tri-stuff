# from ultralytics import YOLO

# model = YOLO("../../Knife-Detection/yolov5/runs/detect/train15/weights/License.pt")
# model.export(format="onnx", dynamic=True)

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("../../Knife-Detection/yolov5/runs/detect/train15/weights/License.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("License.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")