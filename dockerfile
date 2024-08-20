FROM python:3.8.19

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y wget

RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY GroundingDINO/ GroundingDINO/
COPY segment-anything/ segment-anything/
COPY util/ util/
COPY image_search_api.py .

RUN pip install --no-cache-dir -e segment-anything
RUN pip install --no-cache-dir -e GroundingDINO

EXPOSE 8000

CMD ["uvicorn", "image_search_api:app", "--host", "0.0.0.0", "--port", "8000"]
