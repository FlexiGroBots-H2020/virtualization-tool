FROM stereolabs/zed:4.0-runtime-cuda12.1-ubuntu22.04

WORKDIR /wd

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

COPY requirements.txt /wd
RUN apt-get update && apt-get install -y python3-dev gcc g++ libopenmpi-dev
RUN pip install pycocotools
RUN pip install -r requirements.txt

ARG COLMAP_GIT_COMMIT=dev
ARG CUDA_ARCHITECTURES=native
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

WORKDIR /wd

COPY virtual3d_pipeline.py /wd
COPY virtual3d_pipeline_utils.py /wd
COPY colmap2nerf.py /wd
COPY svo_export.py /wd
COPY background_substraction.py /wd
COPY X_Decoder /wd/X_Decoder/
COPY xdcoder_utils.py /wd

RUN rm -rf /usr/local/lib/python3.10/dist-packages/numpy*
RUN pip install numpy==1.25.1

RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


RUN apt-get update && apt-get install -y ffmpeg 

RUN apt-get update && apt-get install -y \
    libx11-xcb1 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libxcb-xkb1

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y curl gnupg

# Configurar el repositorio de paquetes de NVIDIA
RUN curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
RUN curl -sL https://nvidia.github.io/nvidia-docker/ubuntu22.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

# Actualizar la lista de paquetes e instalar nvidia-container-toolkit
RUN apt-get update && apt-get install -y nvidia-container-toolkit

# Resto de tu Dockerfile...
RUN apt-get update && apt-get install -y xvfb

RUN chmod -R 777 /wd

ENTRYPOINT ["python3", "-u", "virtual3d_pipeline.py"]
