import onnxruntime as ort
import numpy as np
import torch
import torchvision
import cv2
import os
confidence_thres = 0.35
iou_thres = 0.5
def postprocess(output): # (self, input_image, output)
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
    #    print("output shape ", output.shape)
    #    outputs = np.transpose(np.squeeze(output))
    #    print("outputs shape ", outputs.shape)
    #    device = torch.device("cpu")
    #    output = torch.from_numpy(output)
    #    print("output shape ", output.shape)
    #    output = output.to(device)
    #    output = output.squeeze().transpose(1,0)
    #    print("output shape ", output.shape)
    #    print(torch.argmax(output[:, 4:], dim=1))
    #    #output = output[torch.argmax(output[:, 4:], dim=1)]
       
    #    output[:, 0] -= output[:, 2]/2
    #    output[:, 1] -= output[:, 3]/2
    #    output[:, 2] += output[:, 0] 
    #    output[:, 3] += output[:, 1] 
    #    # output[:, 3] = -output[:, 3]
    #    #output = output[output[:,4]>0.15]
    #    max_scores, _ = torch.max(output[:, 4:], dim=1)
    #    mask = max_scores > 0.000000001
    #    output = output[mask]
    #    print("output shape ", output.shape)        
    #    # output[:, 0], output[:, 1], output[:, 2], output[:, 3] = output[:, 1], output[:, 3], output[:, 0], output[:, 2] 
    #    output = output[torchvision.ops.nms(output[:, :4],output[:, 4], 0.5)]
    #    print(output.shape)
    #    output[:,[3,1]] = output[:,[1,3]] 
    #    # output[:, 3], output[:, 1] = output[:, 1], output[:, 3]
    #    #return [output[:,:4], output[:,4]]
    #    print("pytorch ", output[:,4].size(dim=0), output[:,:4].type(torch.int32), output[:,4] )
     
       # Get the number of rows in the outputs array
       outs= output
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       print("output earlier shape", output.shape)
       # Convert to torch and move to GPU
       output = torch.from_numpy(output).to(device)  # [N, 4 + num_classes]
       output = output.squeeze().transpose(0, 1) if output.ndim == 3 else output
       print(output)
       print("output shape ", output.shape)
       # Get class scores and corresponding predicted class
       class_scores, class_ids = torch.max(output[:, 4:], dim=1)
       ans=0
       print(class_ids)
       for i in range(len(class_ids)):
           if class_ids[i] >= 2 and class_ids[i] <= 5:
               ans+=1
       print("ans is ", ans)   
       # Filter by confidence threshold
       mask = class_scores > confidence_thres  
       #mask = class_scores > self.confidence_thres  
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
       #print(output)
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
       keep = torchvision.ops.nms(boxes, class_scores,0.5)
       final_boxes = boxes[keep].detach().cpu().numpy().astype(int)
       final_scores = class_scores[keep].detach().cpu().numpy()
       final_class_ids = class_ids[keep].detach().cpu().numpy()  
       print(final_boxes.shape, final_scores.shape, final_class_ids.shape)
       print("final boxes", final_boxes)
       print("final scores", final_scores)
       print("final class ids", final_class_ids)  
       return [np.array(final_boxes), np.array(final_scores), np.array(final_class_ids)]
       #print(outs)    
       outs=outs.T
       outputs=outs
       rows = outs.shape[0]
       #print("rows ", rows)
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
           if max_score > 1e-9:
               # Get the class ID with the highest score
               class_id = np.argmax(classes_scores)
               
    
               # Extract the bounding box coordinates from the current row
               x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
    
               # Calculate the scaled coordinates of the bounding box
               left = int((x - (w / 2)) ) #* x_factor)
               top = int((y - (h / 2)) ) #* y_factor)
               right = int((x + (w / 2)) ) #* x_factor)
               bottom = int((y + (h / 2)) ) # * y_factor)
    
               # Add the class ID, score, and box coordinates to the respective lists
               class_ids.append(class_id)
               scores.append(max_score)
               boxes.append([left, top, right, bottom])
    
       # Apply non-maximum suppression to filter out overlapping bounding boxes
       x = x1
       y = y1
       w = x2 - x1
       h = y2 - y1
       boxes_cv2 = torch.stack([x, y, w, h], dim=1).detach().cpu().numpy()
       boxes_cv2 = boxes_cv2.astype(int).tolist()
       scores = [float(s) for s in scores]  # ensure float
       
       indices = cv2.dnn.NMSBoxes(boxes_cv2, scores, 1e-9, 0.5)
       
       bbox = []
       sscore = []
       
       if len(indices) > 0:
           for i in indices.flatten():
               bbox.append(boxes_cv2[i])
               sscore.append(scores[i])
               #print(class_ids[i])
       
       print("old shapes", [np.array(bbox).shape, np.array(sscore).shape])
       print("new shapes", [np.array(final_boxes).shape, np.array(final_scores).shape])
       print("old ", [np.array(bbox), np.array(sscore)])
       print("new", [final_boxes, final_scores, final_class_ids])
           # # arr.append([box, score])
           # bbox.append(box)
           # sscore.append(score)
    #    print("old shapes", [np.array(bbox).shape, np.array(sscore).shape])
    #    print("new shapes", [np.array(final_boxes).shape, np.array(final_scores).shape])
    #    print("old ", [np.array(bbox),np.array(sscore)])
    #    print("new",[final_boxes, final_scores])
       return [np.array(bbox),np.array(sscore)]
session = ort.InferenceSession("model-repo/License-Detection/1/model.onnx", providers=['CPUExecutionProvider'])
for input_tensor in session.get_inputs():
    print("Name:", input_tensor.name)
    print("Shape:", input_tensor.shape)
    print("Data type:", input_tensor.type)
for output_tensor in session.get_outputs():
    print("Name:", output_tensor.name)
    print("Shape:", output_tensor.shape)
    print("Type:", output_tensor.type)
# dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
# office_image = cv2.imread("../car1.jpg")
# if office_image is None:
#     print("Failed to load image.")
# office_image=(cv2.cvtColor(office_image, cv2.COLOR_BGR2RGB))
# office_image=office_image/255.0
# office_image = cv2.resize(office_image, (640,640))
# office_image = np.expand_dims(office_image,0)
# office_image=office_image.transpose(0,3,1,2)
# print(office_image.shape)
# office_image = office_image.astype(np.float32) 
# image_path = "../car1.jpg"
# save_dir = "./preprocessed_images"
# os.makedirs(save_dir, exist_ok=True)
# base_name = os.path.basename(image_path).split('.')[0]
# np.save(os.path.join(save_dir, f"{base_name}.npy"), office_image)
# # Ensure the image is in float32 format
# outputs = session.run(None, {"images": office_image})
# #print(outputs)# outputs is a list
# print("output shape",len(outputs))
# print("Outputs:", outputs[0].shape)
# postprocess(outputs[0])
#outputs = session.run(None, {"images": dummy_input})  # outputs is a list
# output_array = outputs[0]  # Get the actual numpy array

# print("Original shape:", output_array.shape)  # (1, 6, 8400)

# # Remove batch dimension and transpose to [8400, 6]
# output_array = np.squeeze(output_array, axis=0)# (6, 8400)
# output_array = output_array.transpose(1, 0)    
# print("Transformed shape:", output_array.shape)  # (8400, 6)
# print("Transformed output array:", output_array)   
# print("Outputs:", outputs[0])
# print("Output shape:", outputs[0].shape)