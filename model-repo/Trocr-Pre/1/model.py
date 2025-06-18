import numpy as np
import cv2
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.orig_h = 1080
        self.orig_w = 1920
        self.input_h = 640
        self.input_w = 640
        self.target_h = 200
        self.target_w = 400

    def execute(self, requests):
        responses = []

        for request in requests:
            # Inputs
            image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()  # [1, 3, 920, 1080], FP32
            bboxes = pb_utils.get_input_tensor_by_name(request, "det_bboxes").as_numpy()  # [N, 4], FP32

            # Convert to HWC uint8
            image_chw = image[0]  # [3, 920, 1080]
            image_hwc = np.transpose(image_chw, (1, 2, 0))  # [920, 1080, 3]
            image_hwc = (image_hwc * 255.0).clip(0, 255).astype(np.uint8)

            crops = []

            for box in bboxes:
                x1, y1, x2, y2 = box.astype(np.float32)

                # Map from 640x640 → 1080x920
                scale_x = self.orig_w / self.input_w  # 1080 / 640
                scale_y = self.orig_h / self.input_h  # 920 / 640

                x1_o = int(np.clip(x1 * scale_x, 0, self.orig_w - 1))
                y1_o = int(np.clip(y1 * scale_y, 0, self.orig_h - 1))
                x2_o = int(np.clip(x2 * scale_x, 0, self.orig_w - 1))
                y2_o = int(np.clip(y2 * scale_y, 0, self.orig_h - 1))

                if x2_o <= x1_o or y2_o <= y1_o:
                    continue

                crop = image_hwc[y1_o:y2_o, x1_o:x2_o]
                crop_resized = cv2.resize(crop, (self.target_w, self.target_h), interpolation=cv2.INTER_CUBIC)


                crop_chw = np.transpose(crop_resized, (2, 0, 1)).astype(np.float32) / 255.0
                crops.append(crop_chw)

            if crops:
                output = np.stack(crops, axis=0).astype(np.float32)
            else:
                output = np.zeros((0, 3, self.target_h, self.target_w), dtype=np.float32)

            out_tensor = pb_utils.Tensor("cropped_plates", output)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
