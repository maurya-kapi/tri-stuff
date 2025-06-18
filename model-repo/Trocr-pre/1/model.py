import triton_python_backend_utils as pb_utils
import numpy as np
import cv2

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            # Inputs
            input_images = pb_utils.get_input_tensor_by_name(request, "input_images").as_numpy()         # [B, 3, 640, 640], FP32
            detection_bboxes = pb_utils.get_input_tensor_by_name(request, "detection_bboxes").as_numpy() # [M, 4], INT32
            det_bboxes = pb_utils.get_input_tensor_by_name(request, "det_bboxes").as_numpy()             # [N, 4], FP32
            bbox_batch_index = pb_utils.get_input_tensor_by_name(request, "bbox_batch_index").as_numpy() # [N], INT32
            # heigh= 
            output_crops = []

            for i in range(det_bboxes.shape[0]):
                # Get LP and associated car
                car_idx = bbox_batch_index[i]
                car_box = detection_bboxes[car_idx].astype(np.int32)      # [x1_c, y1_c, x2_c, y2_c]
                lp_box = det_bboxes[i].astype(np.float32)                 # [x1_lp, y1_lp, x2_lp, y2_lp] (in resized 640x640 car image)
                print(f"Processing LP {i} for car {car_idx}: car_box={car_box}, lp_box={lp_box}")
                # Compute car width and height
                x1_c, y1_c, x2_c, y2_c = car_box
                x1_c, y1_c, x2_c, y2_c = map(int, [x1_c, y1_c, x2_c, y2_c])
                W = x2_c - x1_c
                H = y2_c - y1_c

                if W <= 0 or H <= 0:
                    print(f"Invalid car box: {car_box}, skipping LP detection.")
                    continue  # skip invalid car boxes

                # Convert LP bbox from resized car back to original image
                lp_x1, lp_y1, lp_x2, lp_y2 = lp_box
                lp_x1_img = x1_c + (lp_x1 / 640.0) * W
                lp_y1_img = y1_c + (lp_y1 / 640.0) * H
                lp_x2_img = x1_c + (lp_x2 / 640.0) * W
                lp_y2_img = y1_c + (lp_y2 / 640.0) * H

                # Final image coordinates
                lp_x1_img = int(np.clip(lp_x1_img, 0, 639))
                lp_y1_img = int(np.clip(lp_y1_img, 0, 639))
                lp_x2_img = int(np.clip(lp_x2_img, 0, 639))
                lp_y2_img = int(np.clip(lp_y2_img, 0, 639))
                print(f"LP box in image coordinates: ({lp_x1_img}, {lp_y1_img}, {lp_x2_img}, {lp_y2_img})")
                if lp_x2_img <= lp_x1_img or lp_y2_img <= lp_y1_img:
                    print(f"Invalid LP box: ({lp_x1_img}, {lp_y1_img}, {lp_x2_img}, {lp_y2_img}), skipping.")
                    continue  # skip invalid LP box

                # Get the corresponding image (assume batch size = 1)
                image = input_images[0]  # shape: [3, 640, 640], FP32 [0, 1]
                image_hwc = np.transpose(image, (1, 2, 0)) * 255.0  # [H, W, C] in uint8 space
                image_hwc = image_hwc.astype(np.uint8)

                # Crop and resize
                lp_crop = image_hwc[lp_y1_img:lp_y2_img, lp_x1_img:lp_x2_img]  # [H, W, 3]
                if lp_crop.shape[0] == 0 or lp_crop.shape[1] == 0:
                    print("WHYYY!!!")
                    lp_crop = np.zeros((100, 200, 3), dtype=np.uint8)
                else:
                    lp_crop = cv2.resize(lp_crop, (100, 50), interpolation=cv2.INTER_LINEAR)

                # Convert back to [C, H, W] in FP32 [0, 1]
                lp_crop_chw = np.transpose(lp_crop, (2, 0, 1)).astype(np.float32) / 255.0
                output_crops.append(lp_crop_chw)

            # Stack and return
            print(f"Found {len(output_crops)} valid LP crops.")
            if output_crops:
                output_np = np.stack(output_crops, axis=0).astype(np.float32) 
            else:
                print("No valid LP crops found, returning empty output.")
                output_np = np.zeros((0, 3, 100, 200), dtype=np.float32)
            # output_np=np.array(output_crops, dtype=np.float32)  # Ensure correct dtype
            # print(f"Output shape: {output_np.shape}, dtype: {output_np.dtype}")
            out_tensor = pb_utils.Tensor("Cropped_Licenses", output_np)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)

        return responses
