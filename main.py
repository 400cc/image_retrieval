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
    
# 한국어 -> 영어 매핑 테이블 정의
category_translation_map = {
    '여성': 'Women',
    '남성': 'Men',
    '의류': 'Clothing',
    '아우터': 'Outerwear',
    '자켓': 'Jacket',
    '코트': 'Coat',
    '트렌치코트': 'Trench Coat',
    '핸드메이드코트': 'Handmade Coat',
    '점퍼': 'Jumper',
    '후드집업': 'Hood Zip-Up',
    '패딩': 'Padding',
    '레더': 'Leather',
    '퍼': 'Fur',
    '베스트': 'Vest',
    '원피스': 'Dress',
    '미니': 'Mini',
    '미디': 'Midi',
    '맥시': 'Maxi',
    '점프수트': 'Jumpsuit',
    '셋업': 'Setup',
    '블라우스': 'Blouse',
    '셔츠': 'Shirt',
    '티셔츠': 'T-Shirt',
    '반팔': 'Short Sleeve',
    '긴팔': 'Long Sleeve',
    '슬리브리스': 'Sleeveless',
    '스웻': 'Sweat',
    '후드': 'Hood',
    '터틀넥': 'Turtleneck',
    '니트': 'Knit',
    '풀오버': 'Pullover',
    '가디건': 'Cardigan',
    '캐시미어': 'Cashmere',
    '스커트': 'Skirt',
    '롱': 'Long',
    '팬츠': 'Pants',
    '스트레이트': 'Straight',
    '와이드': 'Wide',
    '스키니': 'Skinny',
    '슬랙스': 'Slacks',
    '치노': 'Chino',
    '트레이닝/조거': 'Training/Jogger',
    '카고': 'Cargo',
    '쇼츠': 'Shorts',
    '레깅스': 'Leggings',
    '데님': 'Denim',
    '스트레이트진': 'Straight Jeans',
    '보이프렌드진': 'Boyfriend Jeans',
    '부츠컷진': 'Bootcut Jeans',
    '크롭진': 'Cropped Jeans',
    '키즈어패럴': 'Kids Apparel',
    '라운지웨어': 'Loungewear',
    '파자마세트': 'Pajama Set',
    '파자마상의': 'Pajama Top',
    '파자마하의': 'Pajama Bottom',
    '파자마원피스': 'Pajama Dress',
    '로브': 'Robe',
    '임부복': 'Maternity Wear',
    '언더웨어': 'Underwear',
    '세트': 'Set',
    '브라렛': 'Bralette',
    '와이어브라': 'Wired Bra',
    '나시탑/캐미솔': 'Tank Top/Camisole',
    '팬티': 'Panties',
    '블레이저': 'Blazer',
    '무스탕': 'Mustang',
    '피케': 'Pique',
    '조거': 'Jogger',
    '드로즈': 'Drawers',
    '트렁크': 'Trunks',
    '니트/스웨터': 'Knit/Sweater',
    '맨투맨': 'Sweatshirt',
    '긴소매': 'Long Sleeve',
    '민소매': 'Sleeveless',
    '블루종': 'Blouson',
    '라이더스': 'Riders',
    '플리스': 'Fleece',
    '트러커': 'Trucker',
    '슈트': 'Suit',
    '슈트팬츠': 'Suit Pants',
    '수트재킷': 'Suit Jacket',
    '루즈': 'Loose',
    '테이퍼드': 'Tapered',
    '캡슐 컬렉션': 'Capsule Collection',
}

# 카테고리 목록을 매핑하는 함수
def translate_category(korean_category_list):
    return [category_translation_map.get(category, category) for category in korean_category_list]

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
        # translated_category = translator.translate(category_name, src='ko', dest='en').text
        category_list = [category.strip() for category in category_name.split(',')]
        translated_category_list = translate_category(category_list)
        translated_category = ', '.join(translated_category_list)
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
