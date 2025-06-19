import json

import dlib
import face_recognition_models
from face_recognition.api import _raw_face_landmarks
import numpy as np
import triton_python_backend_utils as pb_utils

from deepface import DeepFace


class TritonPythonModel:

    def initialize(self, args):
        
        self.model_config = model_config = json.loads(args["model_config"])

        face_recognition_model = face_recognition_models.face_recognition_model_location()
        self.face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

        embeddings_config = pb_utils.get_output_config_by_name(
            model_config, "embeddings"
        )

        self.embeddings_dtype = pb_utils.triton_string_to_numpy(
            embeddings_config["data_type"]
        )

        boxes_config = pb_utils.get_output_config_by_name(
            model_config, "detection_bboxes"
        )

        self.boxes_dtype = pb_utils.triton_string_to_numpy(
            boxes_config["data_type"]
        )


    def execute(self, requests):

        responses = []
        for request in requests:
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            # face_locations_tensor = pb_utils.get_input_tensor_by_name(request, "face_locations")
            frame = image_tensor.as_numpy()
            # face_locations = face_locations_tensor.as_numpy()
            try:
              rprsnt = DeepFace.represent(frame, model_name="ArcFace", normalization = "ArcFace")
              embeddings = [x["embedding"] for x in rprsnt]
              boxes = [[x["facial_area"]["x"], x["facial_area"]["y"], x["facial_area"]["w"], x["facial_area"]["h"]] for x in rprsnt]
            except:
              embeddings = [np.ones(512)]
              boxes = [[1,1,1,1]]
            print(np.array(boxes).astype(self.boxes_dtype))

            #embeddings = self.get_batch_encodings(frame, face_locations)

            embeddings_tensor = pb_utils.Tensor(
                "embeddings", np.array(embeddings).astype(self.embeddings_dtype)
            )
            boxes_tensor = pb_utils.Tensor(
                "detection_bboxes", np.array(boxes).astype(self.boxes_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    embeddings_tensor,
                    boxes_tensor
                ]
            )
            responses.append(inference_response)

        return responses
    
