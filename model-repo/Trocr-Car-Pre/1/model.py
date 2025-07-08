import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
import numpy as np
class TritonPythonModel:
    def initialize(self, args):
        # Target original image dimensions
        self.orig_h = 1080
        self.orig_w = 1920
        self.input_h = 640
        self.input_w = 640

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get inputs
            image_np = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()  # [1,3,1080,1920]
            bboxes_np = pb_utils.get_input_tensor_by_name(request, "det_bboxes").as_numpy() # [N,4] in 640x640

            # To torch + GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image = torch.from_numpy(image_np).to(device).squeeze(0) # [3,1080,1920]
            bboxes = torch.from_numpy(bboxes_np).float().to(device)  # [N,4]

            # Scale bbox coordinates from 640 → original
            scale_x = self.orig_w / self.input_w  # 1920/640 = 3.0
            scale_y = self.orig_h / self.input_h  # 1080/640 = 1.6875

            bboxes[:,0] *= scale_x
            bboxes[:,2] *= scale_x
            bboxes[:,1] *= scale_y
            bboxes[:,3] *= scale_y

            # Clamp
            bboxes[:,0] = torch.clamp(bboxes[:,0].round(), 0, self.orig_w - 1)
            bboxes[:,2] = torch.clamp(bboxes[:,2].round(), 0, self.orig_w - 1)
            bboxes[:,1] = torch.clamp(bboxes[:,1].round(), 0, self.orig_h - 1)
            bboxes[:,3] = torch.clamp(bboxes[:,3].round(), 0, self.orig_h - 1)

            crops = []
            heights, widths = [], []

            for box in bboxes:
                x1, y1, x2, y2 = box.long()
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[:, y1:y2, x1:x2]
                h, w = crop.shape[1:]
                crops.append(crop)
                heights.append(h)
                widths.append(w)

            if not crops:
                output = torch.zeros((0,3,1,1), device=device, dtype=torch.float32)
            else:
                max_h = max(heights)
                max_w = max(widths)

                padded_crops = []
                for crop in crops:
                    c, h, w = crop.shape
                    pad_h = max_h - h
                    pad_w = max_w - w
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    padded = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), value=0)
                    padded_crops.append(padded)

                output = torch.stack(padded_crops, dim=0) # [N,3,max_h,max_w]
            mapped_bboxes_np = bboxes.cpu().numpy().astype(np.int32)
            mapped_bboxes_tensor=pb_utils.Tensor("mapped_boxes",mapped_bboxes_np)
            output_np = output.cpu().numpy()
            out_tensor = pb_utils.Tensor("cropped_vehicles", output_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor,mapped_bboxes_tensor]))

        return responses
