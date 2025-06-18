import torch
import torch.nn as nn
import onnx
from ultralytics.nn.tasks import DetectionModel  # or from yolov5 import DetectionModel, depending on your setup
torch.serialization.add_safe_globals([DetectionModel])
# ✅ Step 1: Load or define your model
# Replace this with your actual model import
# Example: model = MyDetectionModel()
model = torch.load("../../Knife-Detection/yolov5/runs/detect/train15/weights/License.pt")  # or wherever you load from
model.eval()

# ✅ Step 2: Create dummy input
dummy_input = torch.randn(1, 3, 640, 640)

# ✅ Step 3: Sanity check the output
with torch.no_grad():
    output = model(dummy_input)

    # If model returns multiple outputs, pick only the final detection output
    if isinstance(output, (tuple, list)):
        output = output[0]  # Adjust as needed
    print("Final output shape:", output.shape)

# ✅ Step 4: Export to ONNX with dynamic batch size
torch.onnx.export(
    model,
    dummy_input,
    "license_detection.onnx",
    input_names=["images"],
    output_names=["output0"],
    dynamic_axes={
        "images": {0: "batch_size"},
        "output0": {0: "batch_size"}
    },
    opset_version=11
)

# ✅ Step 5 (Optional): Verify the ONNX model
onnx_model = onnx.load("license_detection.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model exported and validated successfully.")
