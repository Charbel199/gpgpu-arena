FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies + GUI libraries
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
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

WORKDIR /workspace/gpgpu-arena

# Default: just open a shell
CMD ["bash"]
