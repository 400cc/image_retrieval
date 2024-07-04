import boto3
import mysql.connector
from googletrans import Translator
import psycopg2
from psycopg2 import extras
from util.extract_image_feature import process_image_and_feature

def s3_connection():
    try:
        # s3 클라이언트 생성
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id="AKIA2DNL3N7N4TJT6ZYA",
            aws_secret_access_key="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",
        )
    except Exception as e:
        print(e)
    else:
        return s3
        
def list_objects_in_prefix(s3, bucket_name, prefix):
    paginator = s3.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix}
    page_iterator = paginator.paginate(**operation_parameters)

    objects = []
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                objects.append(obj['Key'])

    return objects

def fetch_cdn_url(file_path, result_list):
    img_urls = []
    for obj_key in result_list:
    # 객체 키에서 bucket_name과 prefix를 제거하여 상대 경로 생성
        relative_path = obj_key
        image_url = file_path + relative_path
        img_urls.append(image_url)
    return img_urls

s3 = s3_connection()
bucket_name = "designovel"
prefix = 'image/'
file_path = "https://cdn.designovel.com/"

result = list_objects_in_prefix(s3, bucket_name, prefix)
cdn_urls = fetch_cdn_url(file_path, result)

# MySQL 데이터베이스 연결 설정
db_host = '127.0.0.1'
db_port = 33308
db_user = 'kwu'
db_password = '32tz6ylkxdgndhz6wmh75vj3'
db_name = 'kwu-devel'

# MySQL 데이터베이스 연결
conn_mysql = mysql.connector.connect(
    host=db_host,
    port=db_port,
    user=db_user,
    password=db_password,
    database=db_name
)

# 커서 생성
cursor = conn_mysql.cursor()

# SQL 쿼리 정의
sql_query = """
    SELECT DISTINCT name
    FROM category
"""

# SQL 쿼리 실행
cursor.execute(sql_query)

# 결과를 리스트로 변환
category_names = [name for (name,) in cursor.fetchall()]

# SQL 쿼리 정의
sql_query = """
    SELECT DISTINCT s.style_id, c.name
    FROM category_style s
    JOIN category c ON c.category_id = s.category_id
"""

# SQL 쿼리 실행
cursor.execute(sql_query)

matrix = cursor.fetchall()

# 결과를 딕셔너리로 변환
mapping_dict = {style_id: name for style_id, name in matrix}

# 커넥션과 커서 닫기
cursor.close()
conn_mysql.close()

# 번역기 객체 생성
translator = Translator()

# 한글을 영어로 번역하여 딕셔너리 생성
translated_dict = {}
for name in category_names:
    result = translator.translate(name, src='ko', dest='en')
    translated_dict[name] = result.text

# mapping_dict의 값을 kor_to_eng 딕셔너리를 사용하여 매핑하기
mapped_dict = {style_id: translated_dict[name] for style_id, name in mapping_dict.items()}

# 매핑된 결과 출력
print(mapped_dict)

# postgresql 연결
conn_pg = psycopg2.connect("dbname=imagevector user=test password=5303 host=localhost")
conn_pg.autocommit = True
cur = conn_pg.cursor()

# 삽입할 데이터를 저장할 리스트
data_to_insert = []

# 각 CDN URL 처리
for i, cdn_url in enumerate(cdn_urls):
    parts = cdn_url.split('/')
    style_id = parts[-2]
    
    # mapping_dict에서 style_id에 해당하는 category를 찾기
    category = mapped_dict.get(style_id, '')

    # 이미지와 피처 처리 함수 호출
    vec = process_image_and_feature(cdn_url, category)
    # vec = vec.cpu().numpy().tolist()
    
    data_to_insert.append((style_id, category, vec, cdn_url))
    print(f'{i}번째 완료')

    # 데이터 베이스에 배치로 삽입
    if len(data_to_insert) >= 100:  # 100개 묶음으로 삽입, 필요에 따라 조절 가능
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO version4 (style_id, category, embedding, cdn_url) 
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (cdn_url) DO NOTHING
        """, data_to_insert)
        conn_pg.commit()
        
        print('100개 아이템 데이터베이스에 삽입 완료')
        data_to_insert = []  # 삽입 후 리스트 초기화
        
# 남은 데이터가 있다면 삽입
if data_to_insert:
    psycopg2.extras.execute_batch(cur, """
        INSERT INTO version4 (style_id, category, embedding, cdn_url) 
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (cdn_url) DO NOTHING
    """, data_to_insert)
    conn_pg.commit()
    print('잔여 데이터 데이터베이스에 삽입 완료')