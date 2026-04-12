FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    python3 \
    python3-pip \
    # OpenGL + X11 for GUI
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    libwayland-dev \
    libxkbcommon-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --break-system-packages triton torch --index-url https://download.pytorch.org/whl/cu130 || \
    echo "triton installation failed. Triton kernels will be skipped."
RUN pip3 install --break-system-packages cuda-tile cupy-cuda13x || \
    echo "cuTile installation failed. cuTile kernels will be skipped."
RUN pip3 install --break-system-packages numpy

WORKDIR /workspace/gpgpu-arena


CMD ["bash"]
