import torch
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda:0")
        self.imgC = 3
        self.imgH = 48
        self.imgW = 320

    def resize_norm_img(self, img, target_shape):
        imgC, imgH, imgW = target_shape
        img = img.unsqueeze(0)  # add batch
        img = torch.nn.functional.interpolate(img, size=(imgH, imgW), mode='bilinear', align_corners=False)
        img = img.squeeze(0)
        return img

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get cropped images directly from previous module
            crops_np = pb_utils.get_input_tensor_by_name(request, "cropped_images").as_numpy()  # shape (N,3,H,W)
            crops_torch = torch.from_numpy(crops_np).float().to(self.device)

            norm_images = []
            width_list = []

            for i in range(crops_torch.shape[0]):
                crop = crops_torch[i] * 255.0  # scale to 0-255
                crop = crop.clamp(0,255)

                # convert to grayscale but keep 3 channels
                gray_crop = crop.mean(0, keepdim=True).repeat(3,1,1)
                width_list.append(gray_crop.shape[2] / float(gray_crop.shape[1]))

                norm_img = self.resize_norm_img(gray_crop, (self.imgC, self.imgH, self.imgW)) / 255.0
                norm_images.append(norm_img.unsqueeze(0))

            # Sort by width/h ratio as you did
            indices = np.argsort(np.array(width_list))

            # Build final batch
            if len(norm_images) == 0:
                norm_img_batch = torch.zeros((1,self.imgC,self.imgH,self.imgW), device=self.device)
            else:
                norm_img_batch = torch.cat([norm_images[i] for i in indices], dim=0)

            # Move to numpy for Triton output
            norm_img_batch_np = norm_img_batch.cpu().numpy().astype(np.float32)
            output_tensor = pb_utils.Tensor("x", norm_img_batch_np)

            responses.append(pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            ))

        return responses
