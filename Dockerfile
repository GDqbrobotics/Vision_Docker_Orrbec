# Use Ubuntu 22.04 as base image
FROM ultralytics/ultralytics:latest

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    python3-dev \
    python3-venv \
    python3-pip \
    python3-opencv \
    libudev-dev \
    ffmpeg \
    udev \
    usbutils \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository and checkout the v2-main branch
RUN git clone https://github.com/orbbec/pyorbbecsdk.git
WORKDIR /ultralytics/pyorbbecsdk
RUN git checkout v2-main

# # Create and activate virtual environment
# RUN python3 -m venv /venv
# ENV PATH="/venv/bin:$PATH"

# Install Python dependencies
RUN pip3 install -r requirements.txt wheel pybind11-stubgen pillow

# Build the project
RUN mkdir build && cd build && \
    cmake -Dpybind11_DIR=$(pybind11-config --cmakedir) .. && \
    make -j$(nproc) && \
    make install

# Set environment variables
ENV PYTHONPATH="$PYTHONPATH:/ultralytics/pyorbbecsdk/install/lib/"
ENV PYTHONPATH="$PYTHONPATH:/usr/local/lib"

# Generate Python stubs
RUN pybind11-stubgen pyorbbecsdk

# Build the wheel package
RUN python3 setup.py bdist_wheel

# Install udev rules
RUN bash ./scripts/install_udev_rules.sh && \
    rm -rf /var/lib/apt/lists/* \
    && pip install paho-mqtt

CMD python3 app/inference.py
