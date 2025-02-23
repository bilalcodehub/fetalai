# Use a base image that aligns with your GPU and CUDA version
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Add metadata for maintainability
LABEL org.opencontainers.image.source=https://github.com/BigDataLab/fastai-docker \
      org.opencontainers.image.description="Docker image for FastAI and related tools with CUDA 12.4 support"

# Set non-interactive mode to prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install all system dependencies in one step
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3 python-is-python3 python3-pip \
    build-essential wget git \
    poppler-utils libgl1-mesa-glx mesa-utils \
    tesseract-ocr libtesseract-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Quarto separately
RUN wget -qO- https://quarto.org/download/latest/quarto-linux-amd64.deb -O quarto.deb && \
    dpkg -i quarto.deb && rm quarto.deb

# Upgrade pip before installing Python libraries
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch, FastAI, and related libraries
RUN pip install --no-cache-dir \
    "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
    fastai timm datasets torcheval accelerate nbdev notebook jupyter jupyterlab ipywidgets jupyter_server segmentation_models_pytorch tabulate opencv-python rich monai && \
    pip install --no-cache-dir \
    langchain langchain-openai pymupdf pdf2image pytesseract openai pillow tiktoken faiss-cpu gdown

# Create a working directory
RUN mkdir -p /data

# Set the working directory
WORKDIR /data

# Expose the port for JupyterLab
EXPOSE 8889

# Set the default command to run JupyterLab without token authentication
ENTRYPOINT ["jupyter", "lab", "--NotebookApp.token=''", "--allow-root", "--ip=0.0.0.0", "--notebook-dir=/data", "--no-browser"]

