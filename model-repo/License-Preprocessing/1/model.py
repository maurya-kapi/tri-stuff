# import triton_python_backend_utils as pb_utils
# from torch.utils.dlpack import to_dlpack,from_dlpack
# import numpy as np
# import torch
# import cv2

# class TritonPythonModel:
#     def initialize(self, args):
#         pass

#     def execute(self, requests):
#         responses = []

#         for request in requests:
#             # Get input tensors
#             input_image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
#             # bbox_tensor = pb_utils.get_input_tensor_by_name(request, "detection_bboxes")
#             # print("Input image is ")
#             # print(input_image_tensor)
#             # Convert to NumPy arrays
#             input_image = input_image_tensor.as_numpy()
#             # bboxes = bbox_tensor.as_numpy() # shape: [B, 4]
#             # print("input images is")
#             # print(input_image)
#             # print("Input image shape:", input_image.shape)
#             # print("Bounding boxes shape:", bboxes.shape)
#             batch_size = input_image.shape[0]
#             cropped_batch = []

#             for i in range(batch_size):
#                 image = input_image[0]  # shape: [3, 640, 640]
#                 # print(image)
#                 # Convert CHW -> HWC for OpenCV
#                 image_hwc = np.transpose(image, (1, 2, 0))

#                 # Crop and resize
#                 # cropped = image_hwc[y1:y2, x1:x2]
#                 resized = cv2.resize(image_hwc, (640, 640), interpolation=cv2.INTER_LINEAR)

#                 # Convert back to CHW
#                 resized_chw = np.transpose(resized, (2, 0, 1))  # shape: [3, 640, 640]
#                 cropped_batch.append(resized_chw)
#                 # print(f"resized image is {resized_chw.shape}")
#                 # print(resized_chw)

#             # Stack into a batch: shape [B, 3, 640, 640]
#             cropped_batch_np = np.stack(cropped_batch, axis=0).astype(np.float32)

#             # Create output tensor
#             out_tensor = pb_utils.Tensor("input_image", cropped_batch_np)
#             inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
#             responses.append(inference_response)

#         return responses
import triton_python_backend_utils as pb_utils
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack
import time
import os

class TritonPythonModel:
    def initialize(self, args):
        self.target_h = 640
        self.target_w = 640
        torch.backends.cudnn.benchmark = True

        # Create log file path
        self.log_file = "/tmp/license_preprocessing_timing.csv"

        # Write CSV header if not already present
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("get_input_tensor,to_dlpack,from_dlpack,interpolate,contiguous,to_pb_tensor,append_response,total\n")

    def execute(self, requests):
        responses = []

        for request in requests:
            times = {}
            start_total = time.perf_counter()

            t0 = time.perf_counter()
            input_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            times["get_input_tensor"] = time.perf_counter() - t0

            t1 = time.perf_counter()
            input_dlpack = input_tensor.to_dlpack()
            times["to_dlpack"] = time.perf_counter() - t1

            t2 = time.perf_counter()
            input_batch = from_dlpack(input_dlpack)
            times["from_dlpack"] = time.perf_counter() - t2

            t3 = time.perf_counter()
            resized_batch = torch.nn.functional.interpolate(
                input_batch,
                size=(self.target_h, self.target_w),
                mode="bilinear",
                align_corners=False
            )
            times["interpolate"] = time.perf_counter() - t3

            t4 = time.perf_counter()
            resized_batch = resized_batch.contiguous()
            times["contiguous"] = time.perf_counter() - t4

            t5 = time.perf_counter()
            out_tensor = pb_utils.Tensor.from_dlpack("input_image", to_dlpack(resized_batch))
            times["to_pb_tensor"] = time.perf_counter() - t5

            t6 = time.perf_counter()
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            times["append_response"] = time.perf_counter() - t6

            times["total"] = time.perf_counter() - start_total

            # Append timing to file (in milliseconds)
            with open(self.log_file, "a") as f:
                f.write(",".join(f"{times[k]*1000:.3f}" for k in [
                    "get_input_tensor", "to_dlpack", "from_dlpack",
                    "interpolate", "contiguous", "to_pb_tensor",
                    "append_response", "total"
                ]) + "\n")

        return responses

# import triton_python_backend_utils as pb_utils
# import torch
# from torch.utils.dlpack import to_dlpack, from_dlpack

# class TritonPythonModel:
#     def initialize(self, args):
#         self.target_h = 640
#         self.target_w = 640
#         torch.backends.cudnn.benchmark = True

#     def execute(self, requests):
#         responses = []

#         for request in requests:
#             # Receive input tensor via DLPack
#             input_tensor = pb_utils.get_input_tensor_by_name(request, "image")
#             input_dlpack = input_tensor.to_dlpack()
#             input_batch = from_dlpack(input_dlpack)  # Shape: [B, 3, 1080, 1920], device: CUDA
#             #print(input_batch)
#             # Resize entire batch at once on GPU
#             resized_batch = torch.nn.functional.interpolate(
#                 input_batch,
#                 size=(self.target_h, self.target_w),
#                 mode="bilinear",
#                 align_corners=False
#             )  # Shape: [B, 3, 640, 640]
#             resized_batch = resized_batch.contiguous()

#             # Return resized batch using DLPack (zero-copy GPU)
#             out_tensor = pb_utils.Tensor.from_dlpack("input_image", to_dlpack(resized_batch))
#             responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

#         return responses
# import triton_python_backend_utils as pb_utils
# import torch
# from torch.utils.dlpack import to_dlpack, from_dlpack

# class TritonPythonModel:
#     def initialize(self, args):
#         self.target_h = 640
#         self.target_w = 640
        
#         # Enable optimizations
#         torch.backends.cudnn.benchmark = True
#         torch.backends.cudnn.allow_tf32 = True
#         torch.backends.cuda.matmul.allow_tf32 = True
        
#         # Pre-allocate tensors for common batch sizes to avoid memory allocation overhead
#         self.device = torch.cuda.current_device()
#         self.preallocated_tensors = {}
#         common_batch_sizes = [1, 2, 4, 8, 16, 32]
        
#         with torch.no_grad():
#             for batch_size in common_batch_sizes:
#                 self.preallocated_tensors[batch_size] = torch.empty(
#                     (batch_size, 3, self.target_h, self.target_w),
#                     dtype=torch.float32,
#                     device=self.device,
#                     memory_format=torch.channels_last
#                 ).contiguous()

#     def execute(self, requests):
#         responses = []
        
#         # Process all requests in a single batch for maximum efficiency
#         if len(requests) > 1:
#             return self._batch_execute(requests)
        
#         with torch.no_grad():
#             for request in requests:
#                 # Get input tensor via DLPack (zero-copy)
#                 input_tensor = pb_utils.get_input_tensor_by_name(request, "image")
#                 input_dlpack = input_tensor.to_dlpack()
#                 input_batch = from_dlpack(input_dlpack)
                
#                 batch_size = input_batch.size(0)
                
#                 # Resize the input batch
#                 resized_batch = torch.nn.functional.interpolate(
#                     input_batch,
#                     size=(self.target_h, self.target_w),
#                     mode="bilinear",
#                     align_corners=False
#                 ).contiguous()
                
#                 # Create output tensor using DLPack (zero-copy)
#                 out_tensor = pb_utils.Tensor.from_dlpack("input_image", to_dlpack(resized_batch))
#                 responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        
#         return responses
    
#     def _batch_execute(self, requests):
#         """Handle multiple requests more efficiently by combining them"""
#         responses = []
        
#         with torch.no_grad():
#             # Collect all input tensors
#             input_batches = []
#             for request in requests:
#                 input_tensor = pb_utils.get_input_tensor_by_name(request, "image")
#                 input_dlpack = input_tensor.to_dlpack()
#                 input_batch = from_dlpack(input_dlpack)
#                 input_batches.append(input_batch)
            
#             # Process each request (they might have different batch sizes)
#             for input_batch in input_batches:
#                 resized_batch = torch.nn.functional.interpolate(
#                     input_batch,
#                     size=(self.target_h, self.target_w),
#                     mode="bilinear",
#                     align_corners=False
#                 ).contiguous()
                
#                 out_tensor = pb_utils.Tensor.from_dlpack("input_image", to_dlpack(resized_batch))
#                 responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        
#         return responses

#     def finalize(self):
#         """Clean up pre-allocated tensors"""
#         self.preallocated_tensors.clear()