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
 
category_mapping_dict = {
    '상의': 'Top',
    '니트/스웨터': 'Knit/Sweater',
    '후드 티셔츠': 'Hood T-Shirt',
    '맨투맨/스웨트셔츠': 'Sweatshirt',
    '긴소매 티셔츠': 'Long Sleeve T-Shirt',
    '셔츠/블라우스': 'Shirt/Blouse',
    '피케/카라 티셔츠': 'Pique/Collar T-Shirt',
    '반소매 티셔츠': 'Short Sleeve T-Shirt',
    '민소매 티셔츠': 'Sleeveless T-Shirt',
    '스포츠 상의': 'Sports Top',
    '기타 상의': 'Other Tops',
    '아우터': 'Outerwear',
    '후드 집업': 'Hood Zip-Up',
    '블루종/MA-1': 'Blouson/MA-1',
    '레더/라이더스 재킷': 'Leather/Riders Jacket',
    '무스탕/퍼': 'Mustang/Fur',
    '트러커 재킷': 'Trucker Jacket',
    '슈트/블레이저 재킷': 'Suit/Blazer Jacket',
    '카디건': 'Cardigan',
    '아노락 재킷': 'Anorak Jacket',
    '플리스/뽀글이': 'Fleece/Sherpa',
    '트레이닝 재킷': 'Training Jacket',
    '스타디움 재킷': 'Stadium Jacket',
    '환절기 코트': 'Transitional Coat',
    '겨울 싱글 코트': 'Winter Single Coat',
    '겨울 더블 코트': 'Winter Double Coat',
    '겨울 기타 코트': 'Other Winter Coat',
    '롱패딩/롱헤비 아우터': 'Long Puffer/Heavy Outerwear',
    '숏패딩 숏헤비 아우터': 'Short Puffer/Heavy Outerwear',
    '패딩 베스트': 'Puffer Vest',
    '베스트': 'Vest',
    '사파리/헌팅 재킷': 'Safari/Hunting Jacket',
    '나일론/코치 재킷': 'Nylon/Coach Jacket',
    '기타 아우터': 'Other Outerwear',
    '바지': 'Pants',
    '데님 팬츠': 'Denim Pants',
    '코튼 팬츠': 'Cotton Pants',
    '슈트 팬츠/슬렉스': 'Suit Pants/Slacks',
    '트레이닝/조거 팬츠': 'Training/Jogger Pants',
    '숏 팬츠': 'Short Pants',
    '레깅스': 'Leggings',
    '점프 슈트/오버울': 'Jumpsuit/Overall',
    '스포츠 하의': 'Sports Bottom',
    '기타 바지': 'Other Pants',
    '원피스': 'Dress',
    '미니 원피스': 'Mini Dress',
    '미디 원피스': 'Midi Dress',
    '맥시 원피스': 'Maxi Dress',
    '스커트': 'Skirt',
    '미니스커트': 'Mini Skirt',
    '미디스커트': 'Midi Skirt',
    '롱스커트': 'Long Skirt',
    '여성': 'Women',
    '남성': 'Men',
    '의류': 'Clothing',
    '자켓': 'Jacket',
    '코트': 'Coat',
    '트렌치코트': 'Trench Coat',
    '핸드메이드코트': 'Handmade Coat',
    '점퍼': 'Jumper',
    '후드집업': 'Hood Zip-Up',
    '패딩': 'Padding',
    '레더': 'Leather',
    '퍼': 'Fur',
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
    '데님': 'Denim',
    '스트레이트진': 'Straight Jeans',
    '보이프렌드진': 'Boyfriend Jeans',
    '부츠컷진': 'Bootcut Jeans',
    '스키니진': 'Skinny Jeans',
    '크롭진': 'Cropped Jeans',
    '키즈어패럴': 'Kids Apparel',
    '하의': 'Bottom',
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
    '드로즈': 'Drawers',
    '트렁크': 'Trunks',
    '블레이저': 'Blazer',
    '데님아우터': 'Denim Outerwear',
    '무스탕': 'Mustang',
    '카라': 'Collar',
    '기타 겨울 코트': 'Other Winter Coat',
    '트렌치 코트': 'Trench Coat',
    '가디건/베스트': 'Cardigan/Vest',
    '다운/패딩': 'Down/Padding',
    '탑': 'Top',
    '스웻셔츠': 'Sweatshirt',
    '드레스': 'Dress',
    '롱/맥시': 'Long/Maxi',
    '펜슬': 'Pencil',
    '플레어': 'Flare',
    '캐주얼': 'Casual',
    '포멀': 'Formal',
    '루즈/테이퍼드': 'Loose/Tapered',
    '슬림/스트레이트': 'Slim/Straight',
    '조거/트랙': 'Jogger/Track',
    '수트': 'Suit',
    '수트재킷': 'Suit Jacket',
    '수트팬츠': 'Suit Pants'
}

translator = Translator()

# 카테고리 목록을 매핑하는 함수
def translate_category(category_name):
    translated_list = []
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
