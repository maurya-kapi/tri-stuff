import torch
import torchvision
import torch.nn as nn

class postprocess(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,output):
    output = output.transpose(2,1)
    output[:, :, 0] -= output[:, :, 2]/2 # x - w/2
    output[:, :, 1] -= output[:, :, 3]/2 # y - h/2
    output[:, :, 2] += output[:, :, 0]   # w + (x-w/2) == x+w/2
    output[:, :, 3] += output[:, :, 1]   # h + (y-h/2) == y+h/2
    # output[:, torch.argmax(output[:, :, 4:], dim=1)!=0, 0] = 0
    mask = torch.full(output.shape, True)
    mask[:,:,4] = output[:, :,4]>0.15
    output[mask] = 0
    bs = output.size(0)
    for b in range(bs):
      # print(torchvision.ops.nms(output[0, :, :4],output[0, :, 4], 0.5))
      print(output[b, torchvision.ops.nms(output[b, :, :4],output[b, :, 4], 0.5)].shape)
    
    output[:, :, [3,1]] = output[:, :, [1,3]]
    return [output[:, :, 4].size(dim=0), output[:, :, :4].type(torch.int32), output[:, :, 4]]

# Create the model by using the above model definition.
torch_model = postprocess()

# Input to the model
x = torch.randn(8, 84, 8400, requires_grad=False)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "model.onnx",              # where to save the model (can be a file or file-like object)
                  # export_params=True,        # store the trained parameter weights inside the model file
                  # opset_version=10,          # the ONNX version to export the model to
                  # do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['INPUT_0'],   # the model's input names
                  output_names = ['num_detections', 'detection_bboxes', 'detection_scores'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'num_detections' : {0 : 'batch_size'},
                                'detection_bboxes' : {0 : 'batch_size'},
                                'detection_scores' : {0 : 'batch_size'}})  



