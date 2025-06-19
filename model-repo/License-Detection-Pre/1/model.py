import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack,from_dlpack
import numpy as np
import torch
import cv2

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get input tensors
            input_image_tensor = pb_utils.get_input_tensor_by_name(request, "input_image")
            bbox_tensor = pb_utils.get_input_tensor_by_name(request, "detection_bboxes")
            print("Input image is ")
            print(input_image_tensor)
            # Convert to NumPy arrays
            input_image = input_image_tensor.as_numpy()
            bboxes = bbox_tensor.as_numpy() # shape: [B, 4]
            print("input images is")
            print(input_image)
            print("Input image shape:", input_image.shape)
            print("Bounding boxes shape:", bboxes.shape)
            batch_size = bboxes.shape[0]
            cropped_batch = []

            for i in range(batch_size):
                image = input_image[0]  # shape: [3, 640, 640]
                x1, y1, x2, y2 = bboxes[i].astype(int)
                print(f"Processing image {i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                print("Image is ")
                print(image)
                # Convert CHW -> HWC for OpenCV
                image_hwc = np.transpose(image, (1, 2, 0))

                # Crop and resize
                cropped = image_hwc[y1:y2, x1:x2]
                resized = cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_LINEAR)

                # Convert back to CHW
                resized_chw = np.transpose(resized, (2, 0, 1))  # shape: [3, 640, 640]
                cropped_batch.append(resized_chw)
                print(f"resized image is {resized_chw.shape}")
                print(resized_chw)

            # Stack into a batch: shape [B, 3, 640, 640]
            cropped_batch_np = np.stack(cropped_batch, axis=0).astype(np.float32)

            # Create output tensor
            out_tensor = pb_utils.Tensor("cropped_image", cropped_batch_np)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses
