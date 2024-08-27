# CUDA 12.1 기반의 Ubuntu 이미지를 사용
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# Python과 pip 설치
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y wget libgl1-mesa-glx

# 모델 파일 다운로드
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 필요한 파일 및 디렉토리 복사
COPY aws.ac.kwu.pem /app/aws.ac.kwu.pem
COPY GroundingDINO/ GroundingDINO/
COPY segment-anything/ segment-anything/
COPY util/ util/
COPY image_search_api.py .

# 프로젝트 내부 설치
RUN pip install --no-cache-dir -e segment-anything
RUN pip install --no-cache-dir -e GroundingDINO

# 포트 노출 및 애플리케이션 실행 명령 설정
EXPOSE 8000
CMD ["uvicorn", "image_search_api:app", "--host", "0.0.0.0", "--port", "8000"]
