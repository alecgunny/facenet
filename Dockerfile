FROM nvcr.io/nvidia/tensorflow:18.04-py3

RUN \
  apt-get update && \
  apt-get install --no-install-recommends -y \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev && \
  rm -rf /var/lib/apt/lists/*

RUN pip install \
  scipy \
  scikit-learn \
  opencv-python \
  h5py \
  matplotlib \
  Pillow \
  requests \
  psutil

RUN \
  mkdir /data && \
  cd /data && \ 
  wget http://vis-www.cs.umass.edu/lfw/lfw.tgz && \
  mkdir -p lfw/raw && \
  tar xvf lfw.tgz -C lfw/raw --strip-components=1

ENV PYTHONPATH /workspace/facenet/src/

WORKDIR /workspace/facenet/

ENTRYPOINT ["python", "-i", "src/align/testing.py"]
