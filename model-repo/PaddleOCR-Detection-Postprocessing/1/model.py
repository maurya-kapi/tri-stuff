import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack,from_dlpack
import numpy as np
import torch
import cv2
import cv2
import numpy as np
import torch
import math

from shapely.geometry import Polygon
import pyclipper
class DBPost:
    """
    DB post processing
    """
    def __init__(
        self,
        thresh=0.2,
        box_thresh=0.5,
        max_candidates=1000,
        unclip_ratio=2.0,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
        **kwargs,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow",
            "fast",
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        # print(num_contours)
        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            # print(contour)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            # print(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio)
            if len(box) > 1:
                continue
            box = np.array(box).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            # print(type(box[:,1]))
            # print(type(dest_height))

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
    def __call__(self, outs_dict, shape_list):
        pred = outs_dict["maps"]
        # print("shape list is:")
        # print(shape_list)
        # if isinstance(pred, paddle.Tensor):
        #     pred = pred.numpy()
        pred = pred[:, 0, :, :]
        # print(np.min(pred))
        # print(np.max(pred))
        # print(np.sum(pred))
        segmentation = pred > self.thresh
        # print(pred)
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel,
                )
            else:
                mask = segmentation[batch_index]
            if self.box_type == "poly":
                boxes, scores = self.polygons_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
            elif self.box_type == "quad":
                boxes, scores = self.boxes_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
            else:
                raise ValueError("box_type can only be one of ['quad', 'poly']")

            boxes_batch.append({"points": boxes})
        return boxes_batch


class TritonPythonModel:
    def initialize(self, args):
        self.dbpost=DBPost()
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
        request = requests[0]

        input_map = pb_utils.get_input_tensor_by_name(request, "fetch_name").as_numpy()
        shape_info = pb_utils.get_input_tensor_by_name(request, "shape_info").as_numpy()
        batch_size = input_map.shape[0]

        inp_dicts = {"maps": input_map}
        # You pass the whole batch (8, 1, H, W) or similar
        outputs = self.dbpost(inp_dicts, shape_info)  # returns list of dicts, length batch_size

        all_outputs = []
        lengths = []
        # print(outputs)
        for output in outputs:
            points = output["points"]  # shape (N_i, 4, 2)
            # print(points)
            # print(points.shape)
            if points.size == 0:
                # Fix empty detections to have shape (0, 4, 2)
                points = np.empty((0, 4, 2), dtype=np.float32)
            all_outputs.append(points)
            lengths.append(points.shape[0])

        flat_output = np.concatenate(all_outputs, axis=0)  # shape (sum_N, 4, 2)
        lengths = np.array(lengths, dtype=np.int32)        # shape (batch_size,)
        flat_output=flat_output.astype(np.float32)
        tensor_points = pb_utils.Tensor("Text_bboxes", flat_output)
        tensor_counts = pb_utils.Tensor("Counts", lengths)

        inference_response = pb_utils.InferenceResponse(output_tensors=[tensor_points, tensor_counts])
        return [inference_response]

        # batch_size = input_image.shape[0]

        # processed_imgs = []
        # processed_shapes = []

        # for i in range(batch_size):
        #     single_image = input_image[i]  # shape (3, H, W)
        #     single_image = single_image.transpose(1, 2, 0)  # (H, W, 3)
        #     single_image = single_image[:, :, ::-1]  # maybe RGB <-> BGR swap
        #     single_image = single_image * 255  # scale back to original range if needed

        #     # Your transform returns (tensor, shape_info)
        #     inp_img_tensor, shape_info = self.transform(single_image)
        #     inp_img_np = inp_img_tensor.detach().cpu().numpy()
        #     shape_info_np = shape_info.detach().cpu().numpy()

        #     processed_imgs.append(inp_img_np)
        #     processed_shapes.append(shape_info_np)

        # # Stack into batched outputs
        # batched_imgs = np.stack(processed_imgs, axis=0)         # shape (batch_size, ...)
        # batched_shapes = np.stack(processed_shapes, axis=0)     # shape (batch_size, ...)

        # # Create output tensors
        # out_tensor_x = pb_utils.Tensor("x", batched_imgs)
        # out_tensor_shape = pb_utils.Tensor("shape_info", batched_shapes)

        # # Single batched response
        # inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_x, out_tensor_shape])
        # responses.append(inference_response)

        return responses

