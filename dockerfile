# FROM python:3.8.19
FROM python:3.9

WORKDIR /app

COPY requirements.txt .

FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04
RUN python3.9 -m pip install --upgrade pip
RUN pip install torch==2.1.0+cu122 torchvision==0.16.0+cu122 torchaudio==2.1.0+cu122 --extra-index-url https://download.pytorch.org/whl/cu122

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y wget libgl1-mesa-glx

RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY aws.ac.kwu.pem /app/aws.ac.kwu.pem

COPY GroundingDINO/ GroundingDINO/
COPY segment-anything/ segment-anything/
COPY util/ util/
COPY image_search_api.py .

RUN pip install --no-cache-dir -e segment-anything
RUN pip install --no-cache-dir -e GroundingDINO

EXPOSE 8000

CMD ["uvicorn", "image_search_api:app", "--host", "0.0.0.0", "--port", "8000"]
