from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io, traceback
import base64
import logging

from util.extract_image_feature import process_image_and_feature_by_app
from util.similarity_search import find_similar_images

# 로깅 설정
logging.basicConfig(level=logging.INFO)  # 로그 레벨 설정 (INFO 이상만 출력)

app = FastAPI()

# 정적 파일 및 템플릿 디렉토리 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 이미지 업로드를 위한 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/process/image')
async def process_image(
    image_upload: UploadFile = File(...), 
    category: str = Form(...),
    top_num: int = Form(5)  # top_num 값을 폼 데이터에서 가져옴
):
    # 파일 확장자 검사
    if not allowed_file(image_upload.filename):
        raise HTTPException(status_code=415, detail="Unsupported file format")

    try:
        # 이미지를 PIL Image로 변환
        image = Image.open(io.BytesIO(await image_upload.read())).convert("RGB")
        
        # 이미지 및 이미지 특징 처리
        segmented_image, image_feature = process_image_and_feature_by_app(image, category)
        
        # 원본 이미지 및 세그먼트된 이미지를 base64로 인코딩
        original_image_byte_array = io.BytesIO()
        segmented_image_byte_array = io.BytesIO()
        
        image.save(original_image_byte_array, format='PNG')
        segmented_image.save(segmented_image_byte_array, format='PNG')
        
        original_image_base64 = base64.b64encode(original_image_byte_array.getvalue()).decode('utf-8')
        segmented_image_base64 = base64.b64encode(segmented_image_byte_array.getvalue()).decode('utf-8')
        
        # 이미지 특징을 JSON 형태로 반환
        image_features_list = [feat.tolist() for feat in image_feature]
        
        # 로그 남기기
        logging.info(f"Processed image: Input data: {category}, Image features: {image_features_list}")
        
        # 유사한 이미지 검색 및 반환
        similar_image_dict = find_similar_images(image_features_list, top_num)
        
        logging.info(f"Similar images: {similar_image_dict}")
        
        return JSONResponse(content={
            "original_image": original_image_base64,
            "segmented_condaimage": segmented_image_base64,
            "image_features": image_features_list,
            "similar_images": similar_image_dict
        })
    
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error processing image")
