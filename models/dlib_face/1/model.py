import json
import time

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


    def execute(self, requests):

        responses = []
        for request in requests:

            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            face_locations_tensor = pb_utils.get_input_tensor_by_name(request, "face_locations")
            frame = image_tensor.as_numpy()
            h,w,_ = frame.shape
            face_locations = face_locations_tensor.as_numpy()
            # np.save("image", frame)
            # embeddings = self.get_batch_encodings(frame, face_locations)
            embeddings = []
            
            for [y,xw,yh,x] in face_locations:
              try:
                #embedding_dict = DeepFace.represent(frame[int(y*0.9):min(h,int(yh*1.1)), int(x*0.9): min(int(xw*1.1))], model_name="ArcFace", normalization = "ArcFace")
                # crop_image = frame[int(y*0.8):int(yh*1.2), int(x*0.8):int(xw*1.2)]
                yhy = abs(y-yh)
                xhx = abs(x-xw)
                crop_image = frame[max(0,int(y-(yhy*0.4))): min(h,int(yh+(yhy*0.1))), max(0,int(x-(xhx*0.3))):min(w,int(xw+(xhx*0.3)))]
                print(x,xw,xhx,int(x-(xhx*0.3)),int(xw+(xhx*0.3)))
                #print(int(y-(yhy*0.3)), int(yh+(yhy*0.3)), yh-y, xw-x)
                embedding_dict = DeepFace.represent(crop_image, model_name="ArcFace", normalization = "ArcFace", detector_backend = "retinaface", align=True )
                # print(embedding_dict[0])
                embedding = embedding_dict[0]["embedding"]  # arcface, remove later
                embeddings.append(embedding)
              except Exception as e:
                  try:
                    print(e)
                    yhy = abs(y-yh)
                    xhx = abs(x-xw)
                    # crop_image = frame[int(y-(yhy*0.5)):int(yh+(yhy*0.1)), int(x-(xhx*0.3)):int(xw+(xhx*0.3))]
                    # crop_image = frame[int(y):int(yh), int(x):int(xw)]
                    embedding_dict_2 = DeepFace.represent(crop_image, model_name="ArcFace", normalization = "ArcFace", enforce_detection=False)
                    print(embedding_dict_2, type(embedding_dict_2))
                    embedding = embedding_dict_2[0]["embedding"]  # arcface, remove later
                    embeddings.append(embedding)
                  except Exception as e:
                      print(e, time.strftime('%X %x %Z'))
                      embeddings.append(np.ones(512))

            # embedding_dict_list = DeepFace.represent(crop_image, model_name="ArcFace", normalization = "ArcFace", detector_backend = "retinaface", align=True )
            # for embedding_dict in embedding_dict_list:
            #   embeddings.append(embedding_dict["embedding"])

            embeddings_tensor = pb_utils.Tensor(
                "embeddings", np.array(embeddings).astype(self.embeddings_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    embeddings_tensor
                ]
            )
            responses.append(inference_response)

        return responses
    
    def get_batch_encodings(self, face_image, known_face_locations, num_jitters=1, model="small"):
        try:
            print("method = get_batch_encodings, status = started")
            dlib_vector = dlib.full_object_detections()
            raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
            dlib_vector.extend(raw_landmarks)
            embeddings = np.array(self.face_encoder.compute_face_descriptor(face_image, dlib_vector, num_jitters))
            print("method = get_batch_encodings, status = completed", embeddings)
            return embeddings
        except Exception as e:
            print(e)
            return np.array([])
