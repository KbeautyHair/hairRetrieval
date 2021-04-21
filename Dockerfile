FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN apt update
RUN apt install libopencv-dev
RUN apt-get install libgl1-mesa-glx
RUN apt-get install libglib2.0-0