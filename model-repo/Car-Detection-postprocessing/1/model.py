import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack,from_dlpack

import cv2
import sys
import numpy as np
import torch
import torchvision

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    def initialize(self, args):
      self.confidence_thres = 0.25
      self.iou_thres = 0.7
      


    def postprocess(self, output):
        # (self, input_image, output)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # Convert to torch and move to GPU
        # output = torch.from_numpy(output).to(device)  # [N, 4 + num_classes]
        # output = output.squeeze()
        # print("output shape ", output.shape)
        # print(output)

        # output = output.transpose(0, 1)
        # print("output shape ", output.shape)
        # print(output)

        # # Get class scores and corresponding predicted class
        # class_scores, class_ids = torch.max(output[:, 4:], dim=1)

        # # Filter by confidence threshold
        # mask = class_scores > self.confidence_thres

        # # Apply class filter: only car (2) and bus (5)
        # allowed_class_ids = [2, 5]
        # class_id_mask = torch.tensor(
        #     [c.item() in allowed_class_ids for c in class_ids],
        #     device=class_ids.device
        # )

        # # Combine both masks
        # final_mask = mask & class_id_mask

        # output = output[final_mask]
        # class_scores = class_scores[final_mask]
        # class_ids = class_ids[final_mask]

        # if output.shape[0] == 0:
        #     return np.array([]), np.array([]), np.array([])
       #outs= output
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Convert to torch and move to GPU
        #output = torch.from_numpy(output).to(device)  # [N, 4 + num_classes]
        print("output earlier shape", output.shape)
        output=output.to(device)  # [N, 4 + num_classes]
        output = output.squeeze()
        print("output shape after squeeze ", output.shape)
        output = output.transpose(1,0)
        print("output shape after transpose ", output.shape) 
        print(output)
        print("output shape ", output.shape)
        # Get class scores and corresponding predicted class
        class_scores, class_ids = torch.max(output[:, 4:], dim=1)   
        print(class_ids)
        #now i will calculate where class_ids is 2 to 5
        ans=0
        for i in range(len(class_ids)):
            if class_ids[i] >= 2 and class_ids[i] <= 5:
                ans+=1
        print("ans is ", ans)
        # Filter by confidence threshold
        mask = class_scores > self.confidence_thres  
        print("LEngth where mask is 1 is ", mask.sum()) 
        # Apply class filter: only car (2) and bus (5)
        allowed_class_ids = torch.tensor([2, 5], device=class_ids.device)
        class_id_mask = torch.isin(class_ids, allowed_class_ids)
        print("length of class_id_mask is", class_id_mask.sum())
         # Combine both masks
        final_mask = mask & class_id_mask
        output = output[final_mask]
        print("final mask shape is")
        print(output.shape)
        print(output)
        class_scores = class_scores[final_mask]
        class_ids = class_ids[final_mask]
        if output.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
    
        # Convert boxes from [cx, cy, w, h] → [x1, y1, x2, y2]
        boxes = output[:, :4].clone()
        x1 = boxes[:, 0] - boxes[:, 2] / 2  # x1
        y1 = boxes[:, 1] - boxes[:, 3] / 2  # y1
        x2 = boxes[:, 0] + boxes[:, 2]      # x2
        y2 = boxes[:, 1] + boxes[:, 3]      # y2
        x1 = x1.clamp(0, 640 - 1)
        y1 = y1.clamp(0, 640 - 1)
        x2 = x2.clamp(0, 640 - 1)
        y2 = y2.clamp(0, 640 - 1)
 
        # Apply NMS
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        keep = torchvision.ops.nms(boxes, class_scores,self.iou_thres)
        final_boxes = boxes[keep].detach().cpu().numpy().astype(int)
        final_scores = class_scores[keep].detach().cpu().numpy()
        final_class_ids = class_ids[keep].detach().cpu().numpy()  
        print(final_boxes.shape, final_scores.shape, final_class_ids.shape)
        return [np.array(final_boxes), np.array(final_scores), np.array(final_class_ids)]
        # # Convert boxes from [cx, cy, w, h] → [x1, y1, x2, y2]
        # boxes = output[:, :4].clone()
        # x1 = boxes[:, 0] - boxes[:, 2] / 2  # x1
        # y1 = boxes[:, 1] - boxes[:, 3] / 2  # y1
        # x2 = boxes[:, 0] + boxes[:, 2]      # x2
        # y2 = boxes[:, 1] + boxes[:, 3]      # y2

        # x1 = x1.clamp(0, 640 - 1)
        # y1 = y1.clamp(0, 640 - 1)
        # x2 = x2.clamp(0, 640 - 1)
        # y2 = y2.clamp(0, 640 - 1)

        # # Apply NMS
        # boxes = torch.stack([x1, y1, x2, y2], dim=1)
        # keep = torchvision.ops.nms(boxes, class_scores, self.iou_thres)

        # final_boxes = boxes[keep].detach().cpu().numpy().astype(int)
        # final_scores = class_scores[keep].detach().cpu().numpy()
        # final_class_ids = class_ids[keep].detach().cpu().numpy()

        # print("final_boxes are", final_boxes)
        # print("final_scores are", final_scores)

        # # Optional: crop images
        # # cropped_images = []
        # # for box in final_boxes:
        # #     x1, y1, x2, y2 = box
        # #     cropped = input_image[y1:y2, x1:x2]
        # #     cropped_images.append(cropped)

        # return [np.array(final_boxes), np.array(final_scores), np.array(final_class_ids)]

       # print("pytorch ", output[:,4].size(dim=0), output[:,:4].type(torch.int32), output[:,4] )
       
    #    # Get the number of rows in the outputs array
    #    rows = outputs.shape[0]
    
    #    # Lists to store the bounding boxes, scores, and class IDs of the detections
    #    boxes = []
    #    scores = []
    #    class_ids = []
    
    #    # Calculate the scaling factors for the bounding box coordinates
    #    # x_factor = self.img_width / self.input_width
    #    # y_factor = self.img_height / self.input_height
    
    #    # Iterate over each row in the outputs array
    #    for i in range(rows):
    #        # Extract the class scores from the current row
    #        classes_scores = outputs[i][4:]
    
    #        # Find the maximum score among the class scores
    #        max_score = np.amax(classes_scores) # np.amax
    
    #        # If the maximum score is above the confidence threshold
    #        if max_score >= self.confidence_thres:
    #            # Get the class ID with the highest score
    #            class_id = np.argmax(classes_scores)
               
    
    #            # Extract the bounding box coordinates from the current row
    #            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
    
    #            # Calculate the scaled coordinates of the bounding box
    #            left = int((x - (w / 2)) ) #* x_factor)
    #            top = int((y + (h / 2)) ) #* y_factor)
    #            right = int((x + (w / 2)) ) #* x_factor)
    #            bottom = int((y - (h / 2)) ) # * y_factor)
    
    #            # Add the class ID, score, and box coordinates to the respective lists
    #            class_ids.append(class_id)
    #            scores.append(max_score)
    #            boxes.append([left, top, right, bottom])
    
    #    # Apply non-maximum suppression to filter out overlapping bounding boxes
    #    indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
    #    print(indices)
    #    arr = []
    #    bbox = []
    #    sscore = []
    #    # Iterate over the selected indices after non-maximum suppression
    #    for i in indices:
    #        # Get the box, score, and class ID corresponding to the index
    #        if class_ids[i]==0:
    #            bbox.append(boxes[i])
    #            sscore.append(scores[i])
    #            class_id = class_ids[i]
    #            print(class_id)
    #        # # arr.append([box, score])
    #        # bbox.append(box)
    #        # sscore.append(score)
    
    #    print("old ", [np.array(bbox),np.array(sscore)])
    #    return [np.array(bbox),np.array(sscore)]


    def execute(self, requests):

        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "output0")
            outputs = from_dlpack(in_0.to_dlpack())
            print("outputs shape", outputs.shape)
            print(outputs)
            #outputs = outputs[0]
            outputs2 = self.postprocess(outputs)
            num_detections = torch.tensor(len(outputs2[0])).type(torch.int32)
            detection_bboxes = torch.from_numpy(outputs2[0]).type(torch.int32)
            detection_scores = torch.from_numpy(outputs2[1]).type(torch.float32)
            # detection_bboxes = outputs2[0].astype(np.float32)
            # detection_scores = outputs2[1].astype(np.float32)
            num_detections = pb_utils.Tensor.from_dlpack("num_detections", to_dlpack(num_detections))
            detection_bboxes = pb_utils.Tensor.from_dlpack("detection_bboxes", to_dlpack(detection_bboxes))
            detection_scores = pb_utils.Tensor.from_dlpack("detection_scores", to_dlpack(detection_scores))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    num_detections,
                    detection_bboxes,
                    detection_scores])

            responses.append(inference_response)
        return responses