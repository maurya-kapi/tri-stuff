import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack,from_dlpack
import numpy as np
import torch
import cv2
import cv2
import numpy as np
import torch
import math

class DetResizeForTestTorch:
    def __init__(self, limit_side_len=960, limit_type="max"):
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type

    def __call__(self, img):
        h, w, _ = img.shape
        img, (ratio_h, ratio_w) = self.resize_image(img, h, w)
        shape_info = np.array([h, w, ratio_h, ratio_w], dtype=np.float32)
        return img, shape_info

    def resize_image(self, img, h, w):
        # Compute ratio based on mode
        if self.limit_type == "max":
            ratio = min(1.0, self.limit_side_len / max(h, w))
        elif self.limit_type == "min":
            ratio = max(1.0, self.limit_side_len / min(h, w))
        elif self.limit_type == "resize_long":
            ratio = float(self.limit_side_len) / max(h, w)
        else:
            raise ValueError(f"Unsupported limit_type: {self.limit_type}")

        # Compute padded sizes
        resize_h = max(32, int(round(h * ratio / 32) * 32))
        resize_w = max(32, int(round(w * ratio / 32) * 32))

        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, (ratio_h, ratio_w)

class NormalizeImageTorch:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean, dtype=np.float32).reshape(1,1,3)
        self.std = np.array(std, dtype=np.float32).reshape(1,1,3)

    def __call__(self, img):
        img = img.astype(np.float32) / 255.0
        return (img - self.mean) / self.std

class ToCHWTorch:
    def __call__(self, img):
        return np.transpose(img, (2, 0, 1))

class ComposeWithShape:
    def __init__(self, limit_side_len=960, limit_type="max"):
        self.resize = DetResizeForTestTorch(limit_side_len, limit_type)
        self.normalize = NormalizeImageTorch()
        self.to_chw = ToCHWTorch()

    def __call__(self, img):
        img, shape_info = self.resize(img)
        img = self.normalize(img)
        img = self.to_chw(img)
        return torch.from_numpy(img), torch.from_numpy(shape_info)

class TritonPythonModel:
    def initialize(self, args):
        self.transform = ComposeWithShape(limit_side_len=960, limit_type="max")
        pass

    # def execute(self, requests):
    #     responses = []

    #     for request in requests:
    #         # Get input tensors
    #         input_image_tensor = pb_utils.get_input_tensor_by_name(request, "cropped_vehicles")
    #         input_image=input_image_tensor.as_numpy()
    #         print("Input image shape is: ",input_image.shape)
    #         # for i in range(input_image.shape[0]):
                
    #         # input_image=input_image.transpose(1,2,0)
    #         # input_image=input_image[:,:,::-1]
    #         # input_image=input_image*255
    #         # inp_img_tensor,shape_info=self.transform(input_image)
    #         # inp_img_np=inp_img_tensor.numpy()
    #         # shape_info=shape_info.numpy()
    #         # inp_img_tensor=pb_utils.Tensor("x", inp_img_np)
    #         # shape_info_tensor=pb_utils.Tensor("shape_info",shape_info)
    #         # inference_response = pb_utils.InferenceResponse(output_tensors=[inp_img_tensor,shape_info_tensor])
    #         # responses.append(inference_response)
    #         batch_size = input_image.shape[0]
    #         processed_imgs = []
    #         processed_shapes = []

    #         for i in range(batch_size):
    #             single_image = input_image[i]  # (3, H, W)
    #             single_image = single_image.transpose(1, 2, 0)  # (H, W, 3)
    #             single_image = single_image[:, :, ::-1]  # maybe RGB <-> BGR
    #             single_image = single_image * 255

    #             inp_img_tensor, shape_info = self.transform(single_image)
    #             inp_img_np = inp_img_tensor.numpy()
    #             shape_info_np = shape_info.numpy()

    #             processed_imgs.append(inp_img_np)
    #             processed_shapes.append(shape_info_np)

    #         # Now stack to make a batch
    #         batched_imgs = np.stack(processed_imgs, axis=0)        # e.g. (8, 3, new_H, new_W) or (8, 4)
    #         batched_shapes = np.stack(processed_shapes, axis=0)    # e.g. (8, 2)

    #         # Make single tensors
    #         inp_img_tensor = pb_utils.Tensor("x", batched_imgs)
    #         shape_info_tensor = pb_utils.Tensor("shape_info", batched_shapes)

    #         # Make single response
    #         inference_response = pb_utils.InferenceResponse(output_tensors=[inp_img_tensor, shape_info_tensor])

    #         # Return single batched response
    #         return [inference_response]

            # bbox_tensor = pb_utils.get_input_tensor_by_name(request, "detection_bboxes")
            # print("Input image is ")
            # print(input_image_tensor)
            # # Convert to NumPy arrays
            # input_image = input_image_tensor.as_numpy()
            # bboxes = bbox_tensor.as_numpy() # shape: [B, 4]
            # print("input images is")
            # print(input_image)
            # print("Input image shape:", input_image.shape)
            # print("Bounding boxes shape:", bboxes.shape)
            # batch_size = bboxes.shape[0]
            # cropped_batch = []

            # for i in range(batch_size):
            #     image = input_image[0]  # shape: [3, 640, 640]
            #     x1, y1, x2, y2 = bboxes[i].astype(int)
            #     print(f"Processing image {i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            #     print("Image is ")
            #     print(image)
            #     # Convert CHW -> HWC for OpenCV
            #     image_hwc = np.transpose(image, (1, 2, 0))

            #     # Crop and resize
            #     cropped = image_hwc[y1:y2, x1:x2]
            #     resized = cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_LINEAR)

            #     # Convert back to CHW
            #     resized_chw = np.transpose(resized, (2, 0, 1))  # shape: [3, 640, 640]
            #     cropped_batch.append(resized_chw)
            #     print(f"resized image is {resized_chw.shape}")
            #     print(resized_chw)

            # # Stack into a batch: shape [B, 3, 640, 640]
            # cropped_batch_np = np.stack(cropped_batch, axis=0).astype(np.float32)

            # Create output tensor
            # out_tensor = pb_utils.Tensor("cropped_image", cropped_batch_np)
            # inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            # responses.append(inference_response)

        # return responses

    def execute(self, requests):
        responses = []

        # Assuming only one request for simplicity, since batching is in first dim
        # In Triton Python backend, if client sends batch_size > 1, you still get one `request`
        request = requests[0]

        # Extract the batched input tensor
        input_image = pb_utils.get_input_tensor_by_name(request, "cropped_vehicles").as_numpy()
        # Shape e.g. (8, 3, H, W)
        batch_size = input_image.shape[0]

        processed_imgs = []
        processed_shapes = []

        for i in range(batch_size):
            single_image = input_image[i]  # shape (3, H, W)
            single_image = single_image.transpose(1, 2, 0)  # (H, W, 3)
            single_image = single_image[:, :, ::-1]  # maybe RGB <-> BGR swap
            single_image = single_image * 255  # scale back to original range if needed

            # Your transform returns (tensor, shape_info)
            inp_img_tensor, shape_info = self.transform(single_image)
            inp_img_np = inp_img_tensor.detach().cpu().numpy()
            shape_info_np = shape_info.detach().cpu().numpy()

            processed_imgs.append(inp_img_np)
            processed_shapes.append(shape_info_np)

        # Stack into batched outputs
        batched_imgs = np.stack(processed_imgs, axis=0)         # shape (batch_size, ...)
        batched_shapes = np.stack(processed_shapes, axis=0)     # shape (batch_size, ...)

        # Create output tensors
        out_tensor_x = pb_utils.Tensor("x", batched_imgs)
        out_tensor_shape = pb_utils.Tensor("shape_info", batched_shapes)

        # Single batched response
        inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_x, out_tensor_shape])
        responses.append(inference_response)

        return responses

