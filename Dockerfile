FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
USER root
# 镜像加速
# COPY ./sources.list /etc/apt/sources.list
RUN apt update && \
    apt install -y wget libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig  python3 pip && \
    wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.23_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.23_amd64.deb && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/log/*
# 如果 libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb 404 not found
# 请打开 http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/ 查找 libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb 对应的新版本
WORKDIR /app
COPY requirements.txt .

# 安装依赖包
RUN pip3 install --no-cache-dir -r requirements.txt   
RUN pip3 install --no-cache-dir ultralytics
RUN pip3 install --no-cache-dir insightface
RUN pip3 install --no-cache-dir onnxruntime
RUN pip3 install --no-cache-dir onnxruntime-gpu

RUN mkdir -p /models/.deepface/weights && \
    wget -nv -O /models/.deepface/weights/retinaface.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5 && \
    wget -nv -O /models/.deepface/weights/facenet512_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5 && \
    wget -nv -O /models/.deepface/weights/arcface_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5

COPY server.py .

ENV API_AUTH_KEY=mt_photos_ai_extra
ENV RECOGNITION_MODEL=buffalo_l
ENV DETECTION_THRESH=0.65
EXPOSE 8000

VOLUME ["/root/.insightface/models"]

CMD [ "python3", "server.py" ]
