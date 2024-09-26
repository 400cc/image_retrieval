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

from pydantic import BaseModel

from util.image_clustering import cluster_and_reduce
from util.extract_image_feature import process_image_and_feature_by_app
from util.similarity_search import find_similar_images

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringRequest(BaseModel):
    mall_type_id: str
    category_list: List[int]
    n_clusters: int


# 이미지 업로드를 위한 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    

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
    category_name: str = Form("apparal"),
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
        
        translator = Translator()
        translated_category = translator.translate(category_name, src='ko', dest='en').text
        print(f'translated_category: {translated_category}')
        # 이미지 및 이미지 특징 처리
        segmented_image, image_feature = process_image_and_feature_by_app(image, translated_category)
        
        # 원본 이미지 및 세그먼트된 이미지를 base64로 인코딩
        # original_image_byte_array = io.BytesIO()
        segmented_image_byte_array = io.BytesIO()
        
        # image.save(original_image_byte_array, format='PNG')
        segmented_image.save(segmented_image_byte_array, format='PNG')
        
        # original_image_base64 = base64.b64encode(original_image_byte_array.getvalue()).decode('utf-8')
        segmented_image_base64 = base64.b64encode(segmented_image_byte_array.getvalue()).decode('utf-8')
        
        # 이미지 특징을 JSON 형태로 반환
        # image_features_list = [feat.tolist() for feat in image_feature]
        
        # 로그 남기기
        # logging.info(f"Processed image: Input data: {category_name}, Image feature: {image_feature}")
        
        # 유사한 이미지 검색 및 반환
        similar_images = find_similar_images(mall_type_id, category_name, category_id_list, image_feature, offset)
        
        logging.info(f"Similar images: {similar_images}")
        
        return JSONResponse(content={
            # "original_image": original_image_base64,
            "segmented_condaimage": segmented_image_base64,
            "similar_images": similar_images
        })
    
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error processing image")
