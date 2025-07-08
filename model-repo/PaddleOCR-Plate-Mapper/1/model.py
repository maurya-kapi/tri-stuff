import numpy as np
import cv2
import triton_python_backend_utils as pb_utils
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack,from_dlpack

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        request = requests[0]

        # Extract inputs from Triton
        vehicle_images = pb_utils.get_input_tensor_by_name(request, "cropped_vehicles").as_numpy()  # (N,3,H,W)
        Text_bboxes = pb_utils.get_input_tensor_by_name(request, "Text_bboxes").as_numpy()        # (sum_counts,4,2)
        Counts = pb_utils.get_input_tensor_by_name(request, "Counts").as_numpy()                 # (N,)
        vehicle_types = pb_utils.get_input_tensor_by_name(request, "detection_class_ids").as_numpy()   # (N,)
        cropped_vehicle_boxes=pb_utils.get_input_tensor_by_name(request,"vehicle_boxes").as_numpy()
        
        num_vehicles = vehicle_images.shape[0]
        final_crops = []
        idx = 0
        final_boxes=[]
        for i in range(num_vehicles):
            num_boxes = Counts[i]
            vehicle_cls = vehicle_types[i]
            vehicle_img = vehicle_images[i].transpose(1,2,0).copy()
            h=vehicle_img.shape[0]
            w=vehicle_img.shape[1]
            scale_x=1920/640
            scale_y=1080/640
            print(f" scale_x is {scale_x} scale_y is {scale_y}")
            cropped_vehicle_boxes[i][0]*=scale_x
            cropped_vehicle_boxes[i][1]*=scale_y
            cropped_vehicle_boxes[i][2]*=scale_x
            cropped_vehicle_boxes[i][3]*=scale_y
            diff_h=abs(abs(cropped_vehicle_boxes[i][1]-cropped_vehicle_boxes[i][3])-h)/2
            diff_w=abs(abs(cropped_vehicle_boxes[i][0]-cropped_vehicle_boxes[i][2])-w)/2
            vehicle_boxes = []
            for _ in range(num_boxes):
                pts = Text_bboxes[idx]
                # xmin = np.min(pts[:, 0])
                # ymin = np.min(pts[:, 1])
                # xmax = np.max(pts[:, 0])
                # ymax = np.max(pts[:, 1])

                # rect_bbox = [xmin, ymin, xmax, ymax]
                
                # mapped_pts=[rect_bbox[0]-diff_w+cropped_vehicle_boxes[i][0],rect_bbox[1]-diff_h+cropped_vehicle_boxes[i][1],rect_bbox[2]-diff_w+cropped_vehicle_boxes[i][0],rect_bbox[3]-diff_h+cropped_vehicle_boxes[i][1]]
                # final_boxes.append(mapped_pts)
                # print(" rectangle bbox is ",rect_bbox)
                # print(" mapped point is ", mapped_pts)
                vehicle_boxes.append(pts)
                idx += 1

            if num_boxes == 0:
                continue

            if vehicle_cls == 0:
                for pts in vehicle_boxes:
                    warp = warp_to_rect(vehicle_img, pts, output_size=(128,32))
                    final_crops.append(warp)
            else:
                boxes_sorted = sorted(vehicle_boxes, key=lambda b: b[0][1])
                prev_pts = boxes_sorted[0]
                prev_warp = warp_to_rect(vehicle_img, prev_pts, output_size=(128,32))

                for next_pts in boxes_sorted[1:]:
                    prev_y_bottom = prev_pts[2][1]
                    prev_x_bottom = prev_pts[2][0]
                    prev_height = abs(prev_pts[0][1] - prev_pts[2][1])
                    prev_width = abs(prev_pts[0][0] - prev_pts[1][0])
                    next_y_top = next_pts[0][1]
                    next_x_top = next_pts[0][0]
                    next_height = abs(next_pts[0][1] - next_pts[2][1])

                    if (next_y_top < prev_y_bottom + 0.5 * prev_height and
                        next_x_top < prev_x_bottom + 0.7 * prev_width and
                        next_x_top > prev_x_bottom - 0.7 * prev_width and
                        abs(next_height - prev_height) < 0.7 * prev_height):
                        
                        next_warp = warp_to_rect(vehicle_img, next_pts, output_size=(128,32))
                        h1, h2 = prev_warp.shape[0], next_warp.shape[0]
                        target_h = max(h1, h2)
                        if h1 < target_h:
                            prev_warp = np.pad(prev_warp, ((0,target_h-h1),(0,0),(0,0)), mode='constant', constant_values=0)
                        if h2 < target_h:
                            next_warp = np.pad(next_warp, ((0,target_h-h2),(0,0),(0,0)), mode='constant', constant_values=0)
                        combined_warp = np.hstack([prev_warp, next_warp])
                        print("combining the images ")
                        # Compute rectangles for both
                        prev_xmin = np.min(prev_pts[:, 0])
                        prev_ymin = np.min(prev_pts[:, 1])
                        prev_xmax = np.max(prev_pts[:, 0])
                        prev_ymax = np.max(prev_pts[:, 1])

                        next_xmin = np.min(next_pts[:, 0])
                        next_ymin = np.min(next_pts[:, 1])
                        next_xmax = np.max(next_pts[:, 0])
                        next_ymax = np.max(next_pts[:, 1])

                        # Now combine these two rects
                        combined_xmin = min(prev_xmin, next_xmin)
                        combined_ymin = min(prev_ymin, next_ymin)
                        combined_xmax = max(prev_xmax, next_xmax)
                        combined_ymax = max(prev_ymax, next_ymax)

                        # Map to original coordinates
                        mapped_combined_pts = [
                            combined_xmin - diff_w + cropped_vehicle_boxes[i][0],
                            combined_ymin - diff_h + cropped_vehicle_boxes[i][1],
                            combined_xmax - diff_w + cropped_vehicle_boxes[i][0],
                            combined_ymax - diff_h + cropped_vehicle_boxes[i][1]
                        ]

                        final_boxes.append(mapped_combined_pts)


                        final_crops.append(combined_warp)
                        prev_pts = next_pts
                        prev_warp = next_warp
                    else:
                        final_crops.append(prev_warp)
                        xmin = np.min(prev_pts[:, 0])
                        ymin = np.min(prev_pts[:, 1])
                        xmax = np.max(prev_pts[:, 0])
                        ymax = np.max(prev_pts[:, 1])

                        rect_bbox = [xmin, ymin, xmax, ymax]
                        
                        mapped_pts=[rect_bbox[0]-diff_w+cropped_vehicle_boxes[i][0],rect_bbox[1]-diff_h+cropped_vehicle_boxes[i][1],rect_bbox[2]-diff_w+cropped_vehicle_boxes[i][0],rect_bbox[3]-diff_h+cropped_vehicle_boxes[i][1]]
                        final_boxes.append(mapped_pts)
                        prev_pts = next_pts
                        prev_warp = warp_to_rect(vehicle_img, next_pts, output_size=(128,32))
                xmin = np.min(prev_pts[:, 0])
                ymin = np.min(prev_pts[:, 1])
                xmax = np.max(prev_pts[:, 0])
                ymax = np.max(prev_pts[:, 1])

                rect_bbox = [xmin, ymin, xmax, ymax]
                        
                mapped_pts=[rect_bbox[0]-diff_w+cropped_vehicle_boxes[i][0],rect_bbox[1]-diff_h+cropped_vehicle_boxes[i][1],rect_bbox[2]-diff_w+cropped_vehicle_boxes[i][0],rect_bbox[3]-diff_h+cropped_vehicle_boxes[i][1]]
                final_boxes.append(mapped_pts)
                final_crops.append(prev_warp)

        if len(final_crops) == 0:
            print("final crops are zeros")
            final_output = np.zeros((1,3,32,32), dtype=np.float32)
            final_boxes=np.zeros((1,4),dtype=np.float32)
        else:
            max_height = max(img.shape[0] for img in final_crops)
            max_width = max(img.shape[1] for img in final_crops)
            batched_imgs = []
            for img in final_crops:
                h, w = img.shape[:2]
                pad_h = max_height - h
                pad_w = max_width - w
                padded_img = np.pad(img, ((0,pad_h), (0,pad_w), (0,0)), mode='constant', constant_values=0)
                batched_imgs.append(padded_img.transpose(2,0,1))  # (3,H,W)
            final_output = np.stack(batched_imgs, axis=0)
        print("out tensor shape is: ",final_output.shape)
        out_tensor = pb_utils.Tensor("cropped_images", final_output)
        final_boxes=np.array(final_boxes,dtype=np.float32)
        final_box_tensor=pb_utils.Tensor("bboxes",final_boxes)
        return [pb_utils.InferenceResponse(output_tensors=[out_tensor,final_box_tensor])]


def warp_to_rect(image, pts, output_size=None):
    pts = pts.astype(np.float32)
    width_top = np.linalg.norm(pts[0] - pts[1])
    width_bottom = np.linalg.norm(pts[3] - pts[2])
    height_left = np.linalg.norm(pts[0] - pts[3])
    height_right = np.linalg.norm(pts[1] - pts[2])
    w = int(max(width_top, width_bottom))
    h = int(max(height_left, height_right))
    if output_size is not None:
        w, h = output_size
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (w,h))
