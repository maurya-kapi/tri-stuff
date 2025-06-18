import numpy as np
import math
import json
import triton_python_backend_utils as pb_utils
import cv2


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        num_detections_config = pb_utils.get_output_config_by_name(
            model_config, "num_detections")
        detection_bboxes_config = pb_utils.get_output_config_by_name(
            model_config, "detection_bboxes")

        detection_scores_config = pb_utils.get_output_config_by_name(
            model_config, "detection_scores")


        # Convert Triton types to numpy types
        self.num_detections_dtype = pb_utils.triton_string_to_numpy(
            num_detections_config['data_type'])

        # Convert Triton types to numpy types
        self.detection_bboxes_dtype = pb_utils.triton_string_to_numpy(
            detection_bboxes_config['data_type'])

        # Convert Triton types to numpy types
        self.detection_scores_dtype = pb_utils.triton_string_to_numpy(
            detection_scores_config['data_type'])

        self.input_height = 640
        self.input_width = 640
        self.conf_threshold = 0.45
        self.iou_threshold = 0.50
        self.reg_max = 16       

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [
            (math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i]))
            for i in range(len(self.strides))
        ]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for (h, w), stride in zip(feats_hw, self.strides):
            x = np.arange(0, w) + grid_cell_offset
            y = np.arange(0, h) + grid_cell_offset
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def process_outputs(self, outputs):
        bboxes, scores = [], []
        for i, pred in enumerate(outputs):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred,
                                      max_shape=(self.input_height, self.input_width)) * stride

            bboxes.append(bbox)
            scores.append(cls)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)

        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences > self.conf_threshold
        bboxes = bboxes[mask]
        confidences = confidences[mask]
        classIds = classIds[mask]

        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold)

        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        if len(indices) > 0:
            mlvl_bboxes = bboxes[indices]
            confidences = confidences[indices]
            return mlvl_bboxes, confidences
        else:
            return np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    @staticmethod
    def softmax(x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        num_detections_dtype = self.num_detections_dtype
        detection_bboxes_dtype = self.detection_bboxes_dtype
        detection_scores_dtype = self.detection_scores_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_397 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            in_389 = pb_utils.get_input_tensor_by_name(request, "INPUT_1")
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_2")

            # Get the output arrays from the results
            outputs = [in_397.as_numpy(), in_389.as_numpy(), in_0.as_numpy()]

            detection_bboxes, detection_scores = self.process_outputs(outputs)
 
            num_detections = np.array(len(detection_bboxes))
            num_detections = pb_utils.Tensor(
                "num_detections", num_detections.astype(num_detections_dtype))

            detection_bboxes = pb_utils.Tensor(
                "detection_bboxes", detection_bboxes.astype(detection_bboxes_dtype))

            detection_scores = pb_utils.Tensor(
                "detection_scores", detection_scores.astype(detection_scores_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    num_detections,
                    detection_bboxes,
                    detection_scores
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass