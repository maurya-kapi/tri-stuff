FROM nvcr.io/nvidia/tritonserver:23.09-py3

RUN pip install opencv-python && \
    apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get install build-essential cmake pkg-config -y

RUN pip install --upgrade pip
RUN pip install face_recognition==1.3.0 dlib  opencv-python numpy
RUN pip install tensorflow[and-cuda]
RUN pip3 install torch torchvision torchaudio 
RUN pip install tensorflow[and-cuda]

RUN apt remove python3-blinker -y
RUN pip install deepface
RUN pip install tf-keras
RUN pip install transformers
CMD ["tritonserver", "--model-repository=/models"]
