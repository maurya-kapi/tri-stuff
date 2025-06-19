from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import triton_python_backend_utils as pb_utils
import numpy as np
import re
import os
class TritonPythonModel:
    # def initialize(self, args):
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.processor = TrOCRProcessor.from_pretrained("model-repo/trocr/1/hf_model/",tokenizer_kwargs={"use_fast": True},use_fast=True)
    #     # self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed",tokenizer_kwargs={"use_fast": True},use_fast=True)
    #     self.model = VisionEncoderDecoderModel.from_pretrained("/home/maurya_patel/tri-stuff/model-repo/trocr/1/hf_model")
    #     #self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(self.device)
    #     self.model.eval()
    def initialize(self, args):
        model_dir = os.path.join(os.path.dirname(__file__), "hf_model")
        print("Resolved path to model:", model_dir)
        print("Files inside model are :",os.listdir(model_dir))
        print("Exists:", os.path.exists(os.path.join(model_dir, "preprocessor_config.json")))
        self.processor = TrOCRProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        self.model.to("cuda")
    def filter_indian_number_plates(self,plates):
        # Regex for: 2 letters + 2 digits + 1-3 letters + 4 digits
        pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')
        valid_plates = [plate for plate in plates if pattern.match(plate)]
        valid_plates=plates
        return valid_plates
    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            input_array = input_tensor.as_numpy()  # shape: [B, 3, H, W] FP32

            batch_texts = []

            for i in range(input_array.shape[0]):
                # Extract one image, convert to uint8 for PIL
                chw_image = input_array[i]  # [3, H, W]
                chw_image = (chw_image * 255).clip(0, 255).astype(np.uint8)
                
                # Convert to HWC
                hwc_image = np.transpose(chw_image, (1, 2, 0))  # [H, W, 3]
                # print(hwc_image)
                pil_image = Image.fromarray(hwc_image, mode="RGB")
                image_array = np.array(pil_image)
                # np.save("i1.npy", image_array)
                # print(f"Processing image {i}")
                # print("saving hwc image")
                # np.save("p.npy", hwc_image)
                # hwc_image = np.transpose(input_array[i], (1, 2, 0))  # shape: [H, W, 3], float32 in [0,1]
                pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
                # print(f"Pixel values shape: {pixel_values.shape}")
                # print("saving pixel values")
                # np.save("pixel_values.npy", pixel_values.cpu().numpy())
                generated_ids = self.model.generate(pixel_values)
                # print(f"Generated IDs for image {i}: {generated_ids}")
                decoded_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                decoded_text = decoded_text.replace(" ", "")
                
                decoded_text = self.filter_indian_number_plates([decoded_text]) 
                #print(f"Decoded text for image {i}: {decoded_text}")
                if(len(decoded_text)>0):
                    decoded_text = decoded_text[0]
                    batch_texts.append(decoded_text.encode("utf-8"))
                else:
                    decoded_text = ""
                # print(f"Decoded text for image {i}: {decoded_text}")
                # batch_texts.append(decoded_text.encode("utf-8"))
            #batch_texts = self.filter_indian_number_plates(batch_texts)
            # Output tensor: shape [B], dtype object (bytes)
            output_tensor = pb_utils.Tensor("OUTPUT_TEXT", np.array(batch_texts, dtype=object))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
