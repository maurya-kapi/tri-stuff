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
      self.confidence_thres = 0.15
      self.iou_thres = 0.5
      


    def postprocess(self, output): # (self, input_image, output)
       """
       Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.
    
       Args:
           input_image (numpy.ndarray): The input image.
           output (numpy.ndarray): The output of the model.
    
       Returns:
           numpy.ndarray: The input image with detections drawn on it.
       """
       # print("1",output.shape)
       # input0 = pb_utils.Tensor.from_dlpack("INPUT0", to_dlpack(pytorch_tensor))
    
       # print("bbb we are using this postprocess") 
       # these poeple have not even given expected output variable type,shape.
       # Transpose and squeeze the output to match the expected shape
       outputs = np.transpose(np.squeeze(output))
       device = torch.device("cpu")
       output = torch.from_numpy(output)
       output = output.to(device)
       output = output.squeeze().transpose(1,0)
       output = output[torch.argmax(output[:, 4:], dim=1)==0]
       output[:, 0] -= output[:, 2]/2
       output[:, 1] -= output[:, 3]/2
       output[:, 2] += output[:, 0] 
       output[:, 3] += output[:, 1] 
       # output[:, 3] = -output[:, 3]
       output = output[output[:,4]>0.15]
       # output[:, 0], output[:, 1], output[:, 2], output[:, 3] = output[:, 1], output[:, 3], output[:, 0], output[:, 2] 
       output = output[torchvision.ops.nms(output[:, :4],output[:, 4], 0.5)]
       output[:,[3,1]] = output[:,[1,3]] 
       # output[:, 3], output[:, 1] = output[:, 1], output[:, 3]
       return [output[:,:4], output[:,4]]
       # print("pytorch ", output[:,4].size(dim=0), output[:,:4].type(torch.int32), output[:,4] )
       
       # Get the number of rows in the outputs array
       rows = outputs.shape[0]
    
       # Lists to store the bounding boxes, scores, and class IDs of the detections
       boxes = []
       scores = []
       class_ids = []
    
       # Calculate the scaling factors for the bounding box coordinates
       # x_factor = self.img_width / self.input_width
       # y_factor = self.img_height / self.input_height
    
       # Iterate over each row in the outputs array
       for i in range(rows):
           # Extract the class scores from the current row
           classes_scores = outputs[i][4:]
    
           # Find the maximum score among the class scores
           max_score = np.amax(classes_scores) # np.amax
    
           # If the maximum score is above the confidence threshold
           if max_score >= self.confidence_thres:
               # Get the class ID with the highest score
               class_id = np.argmax(classes_scores)
               
    
               # Extract the bounding box coordinates from the current row
               x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
    
               # Calculate the scaled coordinates of the bounding box
               left = int((x - (w / 2)) ) #* x_factor)
               top = int((y + (h / 2)) ) #* y_factor)
               right = int((x + (w / 2)) ) #* x_factor)
               bottom = int((y - (h / 2)) ) # * y_factor)
    
               # Add the class ID, score, and box coordinates to the respective lists
               class_ids.append(class_id)
               scores.append(max_score)
               boxes.append([left, top, right, bottom])
    
       # Apply non-maximum suppression to filter out overlapping bounding boxes
       indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
       print(indices)
       arr = []
       bbox = []
       sscore = []
       # Iterate over the selected indices after non-maximum suppression
       for i in indices:
           # Get the box, score, and class ID corresponding to the index
           if class_ids[i]==0:
               bbox.append(boxes[i])
               sscore.append(scores[i])
               class_id = class_ids[i]
               print(class_id)
           # # arr.append([box, score])
           # bbox.append(box)
           # sscore.append(score)
    
       print("old ", [np.array(bbox),np.array(sscore)])
       return [np.array(bbox),np.array(sscore)]


    def execute(self, requests):

        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            outputs = from_dlpack(in_0.to_dlpack())
            outputs2 = self.postprocess(outputs.cpu().detach().numpy())
            num_detections = torch.tensor(len(outputs2[0])).type(torch.int32)
            # detection_bboxes = torch.from_numpy(outputs2[0]).type(torch.float32)
            # detection_scores = torch.from_numpy(outputs2[1]).type(torch.float32)
            detection_bboxes = outputs2[0].type(torch.float32)
            detection_scores = outputs2[1].type(torch.float32)
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

