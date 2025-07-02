# import numpy as np
# import cv2
# import triton_python_backend_utils as pb_utils
# def resize_norm_img_svtr(img, image_shape):
#     imgC, imgH, imgW = image_shape
#     resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
#     resized_image = resized_image.astype("float32")
#     resized_image = resized_image.transpose((2, 0, 1)) / 255
#     # resized_image -= 0.5
#     # resized_image /= 0.5
#     return resized_image
# class TritonPythonModel:
#     def initialize(self, args):
#         self.orig_h = 1080
#         self.orig_w = 1920
#         self.input_h = 640
#         self.input_w = 640
#         self.target_h = 200
#         self.target_w = 400
#         self.imgC=3
#         self.imgH=48
#         self.imgW=320

#     def execute(self, requests):
#         responses = []

#         for request in requests:
#             # Inputs
#             image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()  # [1, 3, 920, 1080], FP32
#             bboxes = pb_utils.get_input_tensor_by_name(request, "det_bboxes").as_numpy()  # [N, 4], FP32
#             print("bboxes are:", bboxes)
#             # Convert to HWC uint8
#             image_chw = image[0]  # [3, 920, 1080]
#             image_hwc = np.transpose(image_chw, (1, 2, 0))  # [920, 1080, 3]
#             image_hwc = (image_hwc * 255.0).clip(0, 255).astype(np.uint8)

#             crops = []
#             width_list=[]
#             img_num= bboxes.shape[0]
            
#             for box in bboxes:
#                 x1, y1, x2, y2 = box.astype(np.float32)

#                 # Map from 640x640 → 1080x920
#                 scale_x = self.orig_w / self.input_w  # 1080 / 640
#                 scale_y = self.orig_h / self.input_h  # 920 / 640

#                 x1_o = int(np.clip(x1 * scale_x, 0, self.orig_w - 1))
#                 y1_o = int(np.clip(y1 * scale_y, 0, self.orig_h - 1))
#                 x2_o = int(np.clip(x2 * scale_x, 0, self.orig_w - 1))
#                 y2_o = int(np.clip(y2 * scale_y, 0, self.orig_h - 1))

#                 if x2_o <= x1_o or y2_o <= y1_o:
#                     continue

#                 crop = image_hwc[y1_o:y2_o, x1_o:x2_o]
# # Convert to grayscale and back to RGB (3-channel)
#                 gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
#                 crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
#                 # crop_chw = np.transpose(crop,(2,0,1))
#                 crops.append(crop)
#                 # print(crop)
#                 width_list.append(crop.shape[1] / float(crop.shape[0]))
#             indices = np.argsort(np.array(width_list))
#             # print("Indices", indices)
#             # print("Width List", width_list)
#             batch_num = indices.shape[0]
            
#             # print("Batch Num", batch_num)
#             max_wh_ratio = 0
#             wh_ratio_list=[]
#             norm_images=[]
#             for i in range(batch_num):
#                 h,w = crops[indices[i]].shape[1], crops[indices[i]].shape[2]
#                 wh_ratio= w*1.0 / h
#                 wh_ratio_list.append(wh_ratio)
#                 max_wh_ratio = max(max_wh_ratio, wh_ratio)
#                 # print("Before Resize")
#                 # print(crops[indices[i]].shape)
#                 norm_img=resize_norm_img_svtr(crops[indices[i]], (self.imgC, self.imgH, self.imgW))
#                 # print("After Resize")
#                 # print(norm_img.shape)
#                 norm_img=norm_img[np.newaxis, :]
#                 norm_images.append(norm_img)
#             if len(norm_images) == 0:
#                 norm_img_batch = np.zeros((1, self.imgC, self.imgH, self.imgW), dtype=np.float32)
#             else:
#                 norm_img_batch = np.concatenate(norm_images)
#             #print("shape of x is ", norm_img_batch.shape)
#             output_tensor = pb_utils.Tensor("x", norm_img_batch)
#             wh_tensor = pb_utils.Tensor("wh_ratio_list", np.array(wh_ratio_list, dtype=np.float32))
#             idx_tensor = pb_utils.Tensor("sorted_indices", np.array(indices, dtype=np.int32))
#             max_wh_tensor = pb_utils.Tensor("max_wh_ratio", np.array([max_wh_ratio], dtype=np.float32))

#             inference_response = pb_utils.InferenceResponse(
#                 output_tensors=[output_tensor, wh_tensor, idx_tensor, max_wh_tensor]
#             )
#             responses.append(inference_response)

#         return response
import torch
import cv2
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda:0")
        self.orig_h = 1080
        self.orig_w = 1920
        self.input_h = 640
        self.input_w = 640
        self.imgC = 3
        self.imgH = 48
        self.imgW = 320

    def resize_norm_img(self, img, target_shape):
        # img: torch tensor (C,H,W), float32 on CUDA
        imgC, imgH, imgW = target_shape
        img = img.unsqueeze(0)  # add batch
        img = torch.nn.functional.interpolate(img, size=(imgH, imgW), mode='bilinear', align_corners=False)
        img = img.squeeze(0)
        return img

    def execute(self, requests):
        responses = []
        for request in requests:
            image_np = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()  # [1,3,H,W]
            bboxes = pb_utils.get_input_tensor_by_name(request, "det_bboxes").as_numpy()  # [N,4]
            
            image_np = image_np[0]
            image_torch = torch.from_numpy(image_np).to(self.device)  # [3,H,W], float32
            image_torch = image_torch * 255.0  # scale back
            image_torch = image_torch.clamp(0,255)

            crops = []
            width_list = []
            
            scale_x = self.orig_w / self.input_w
            scale_y = self.orig_h / self.input_h

            for box in bboxes:
                x1, y1, x2, y2 = box.astype(np.float32)
                x1_o = int(np.clip(x1 * scale_x, 0, self.orig_w - 1))
                y1_o = int(np.clip(y1 * scale_y, 0, self.orig_h - 1))
                x2_o = int(np.clip(x2 * scale_x, 0, self.orig_w - 1))
                y2_o = int(np.clip(y2 * scale_y, 0, self.orig_h - 1))

                if x2_o <= x1_o or y2_o <= y1_o:
                    continue

                crop = image_torch[:, y1_o:y2_o, x1_o:x2_o]  # still on GPU
                # grayscale by averaging channels
                gray_crop = crop.mean(0, keepdim=True).repeat(3,1,1)
                
                crops.append(gray_crop)
                width_list.append(gray_crop.shape[2] / float(gray_crop.shape[1]))
            
            indices = np.argsort(np.array(width_list))
            norm_images = []
            # wh_ratio_list = []
            # max_wh_ratio = 0

            for i in range(len(indices)):
                crop = crops[indices[i]]
                h, w = crop.shape[1:]
                # wh_ratio = w * 1.0 / h
                # wh_ratio_list.append(wh_ratio)
                # max_wh_ratio = max(max_wh_ratio, wh_ratio)
                norm_img = self.resize_norm_img(crop, (self.imgC, self.imgH, self.imgW)) / 255.0
                norm_images.append(norm_img.unsqueeze(0))
            
            if len(norm_images) == 0:
                norm_img_batch = torch.zeros((1,self.imgC,self.imgH,self.imgW), device=self.device)
            else:
                norm_img_batch = torch.cat(norm_images, dim=0)

            # move to CPU numpy for Triton output
            norm_img_batch_np = norm_img_batch.cpu().numpy().astype(np.float32)
            output_tensor = pb_utils.Tensor("x", norm_img_batch_np)
            # wh_tensor = pb_utils.Tensor("wh_ratio_list", np.array(wh_ratio_list, dtype=np.float32))
            # idx_tensor = pb_utils.Tensor("sorted_indices", np.array(indices, dtype=np.int32))
            # max_wh_tensor = pb_utils.Tensor("max_wh_ratio", np.array([max_wh_ratio], dtype=np.float32))
            
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            ))

        return responses
