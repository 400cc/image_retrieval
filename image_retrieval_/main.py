from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from PIL import Image
import io, traceback
import json
import base64
import logging
from googletrans import Translator
import os
import torch
import time
from contextlib import asynccontextmanager
from pydantic import BaseModel

from util.image_clustering import cluster_and_reduce
# from util.extract_image_feature import process_image_and_feature_by_app
from util.extract_image_feature import extractImageFeature
from util.similarity_search import find_similar_images

CATEGORY_MAPPING_FILE = "util/category_mapping.json"

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = os.getenv("device", "cuda:0")
logging.info(f"SELECTED DEVICE: {device}")

# GPU 준비 상태를 위해 전역 변수로 embedding 객체를 미리 정의
embedding = extractImageFeature(device=device)

class ClusteringRequest(BaseModel):
    mall_type_id: str
    category_list: List[int]
    n_clusters: int


# 이미지 업로드를 위한 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_category_mapping():
    with open(CATEGORY_MAPPING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


translator = Translator()
        

# 카테고리 목록을 매핑하는 함수
def translate_category(category_name):
    translated_list = []
    category_mapping_dict = load_category_mapping()
    category_list = [category.strip().strip('"') for category in category_name.split(",")]
    for category in category_list:
        # 카테고리 매핑 딕셔너리에 있는 경우 매핑 사용
        if category in category_mapping_dict:
            translated_list.append(category_mapping_dict[category])
        else:
            # 매핑이 없는 경우 Translator를 사용해 번역
            translated_category = translator.translate(category, src='ko', dest='en').text
            translated_list.append(translated_category)
    
    return translated_list

@app.post("/clustering")
def process_clustering(request: ClusteringRequest):
    try:
        logger.info(f"Received clustering request: {request}")
        data_points = cluster_and_reduce(request.n_clusters, request.mall_type_id, request.category_list)
        return {"data_points": data_points}
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/process/image')
async def process_image(
    image_upload: UploadFile = File(...), 
    category_name: str = Form("apparel"),
    offset: int = Form(5),
    category_id_list: str = Form(""),
    mall_type_id: str = Form(None)
):
    
    category_id_list = json.loads(category_id_list)

    # 파일 확장자 검사
    if not allowed_file(image_upload.filename):
        raise HTTPException(status_code=415, detail="Unsupported file format")

    try:
        # 이미지를 PIL Image로 변환
        image = Image.open(io.BytesIO(await image_upload.read())).convert("RGB")

        translated_categories = translate_category(category_name)
        translated_category = ",".join(translated_categories)
        print(f'translated_category: {translated_category}')

        segmented_image, image_feature = embedding.process_image_and_feature_by_app(image, translated_category)

        segmented_image_byte_array = io.BytesIO()
        
        segmented_image.save(segmented_image_byte_array, format='PNG')
        
        segmented_image_base64 = base64.b64encode(segmented_image_byte_array.getvalue()).decode('utf-8')

        # 유사한 이미지 검색 및 반환
        similar_images = find_similar_images(mall_type_id, category_name, category_id_list, image_feature, offset)
        
        
        return JSONResponse(content={
            # "original_image": original_image_base64,
            "segmented_condaimage": segmented_image_base64,
            "similar_images": similar_images
        })
    
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error processing image")
