# FROM python:3.8.19
# FROM python:3.9
FROM nvidia/cuda:12.3.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libpq-dev \
    libboost-all-dev \
    ninja-build \
    wget \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

RUN python3.10 -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu123

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y wget libgl1-mesa-glx

RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY aws.ac.kwu.pem /app/aws.ac.kwu.pem

COPY GroundingDINO/ GroundingDINO/
COPY segment-anything/ segment-anything/
COPY util/ util/
COPY image_search_api.py .

WORKDIR /app/GroundingDINO
RUN python3 setup.py build_ext --inplace

RUN pip install --no-cache-dir -e segment-anything
RUN pip install --no-cache-dir -e GroundingDINO

EXPOSE 8000

CMD ["uvicorn", "image_search_api:app", "--host", "0.0.0.0", "--port", "8000"]
