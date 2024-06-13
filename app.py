from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tempfile import NamedTemporaryFile
import os
from PIL import Image
import io
import base64
import logging

from image_search import process_image_and_feature  # 이미지 처리 함수 import

# 로깅 설정
logging.basicConfig(level=logging.INFO)  # 로그 레벨 설정 (INFO 이상만 출력)

app = FastAPI()

# 정적 파일 및 템플릿 디렉토리 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 이미지 업로드를 위한 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/process_image')
async def process_image(file: UploadFile = File(...), input_data: str = Form(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # 이미지 저장
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(filepath, 'wb') as f:
        f.write(file.file.read())
    
    # 로그 남기기
    logging.info(f"Uploaded file: {file.filename}, Input data: {input_data}")
    
    # 이미지 및 이미지 특징 처리
    image, image_feature = process_image_and_feature(filepath, input_data)
    
    # 이미지를 base64로 인코딩하여 전송
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()
    img_base64 = base64.b64encode(img_byte_array).decode('utf-8')
    
    # 이미지 특징을 JSON 형태로 반환
    image_features_list = [feat.tolist() for feat in image_feature]
    
    # 로그 남기기
    logging.info(f"Processed image: {file.filename}, Image features: {image_features_list}")
    
    return {"image": img_base64, "image_features": image_features_list}
