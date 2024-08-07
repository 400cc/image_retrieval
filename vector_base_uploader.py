import boto3
import mysql.connector
from googletrans import Translator
import psycopg2
from typing import List
from psycopg2 import extras
import paramiko
from sshtunnel import SSHTunnelForwarder
from util.extract_image_feature import process_image_and_feature
from util.mysql_db_util import get_db_connection, create_connection_pool
from load_category_hierarchy import get_category_hierarchy, load_category_hierarchy
import traceback

config_path = 'util/mysql_config.json'
connection_pool = create_connection_pool(config_path)


def load_category_names():
    conn = get_db_connection(connection_pool)
    # 커서 생성
    cursor = conn.cursor()

    # SQL 쿼리 정의
    sql_query = """
        SELECT DISTINCT name
        FROM category
    """

    # SQL 쿼리 실행
    cursor.execute(sql_query)

    # 결과를 리스트로 변환
    category_names = [name for (name,) in cursor.fetchall()]
    conn.close()
    return category_names

def translate_category_name(category_names):
    # 번역기 객체 생성
    translator = Translator()

    # 한글을 영어로 번역하여 딕셔너리 생성
    translated_dict = {}
    for name in category_names:
        result = translator.translate(name, src='ko', dest='en')
        translated_dict[name] = result.text
        
    return translated_dict

def mapping_translated_category(translated_dict):
    category_hierarchy = load_category_hierarchy()
    translated_category_hierarchy = {
        style_id: [[translated_dict[category] for category in sublist] for sublist in categories]
        for style_id, categories in category_hierarchy.items()
    }
    return translated_category_hierarchy


def fetch_cdn_urls(batch_size: int = 1000):
    conn = get_db_connection(connection_pool)
    cursor = conn.cursor()
    
    offset = 0
    all_cdn_urls = []
    while True:
        query = f"SELECT url FROM image LIMIT {batch_size} OFFSET {offset}"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            break
        
        all_cdn_urls.extend([row[0] for row in rows])
        offset += batch_size
        print(f"{offset}개의 URL을 불러왔습니다.")
    
    cursor.close()
    conn.close()
    
    return all_cdn_urls

def get_pg_connection():
    ssh_host = '54.180.146.236'
    ssh_port = 22
    ssh_user = 'ubuntu'
    ssh_private_key = "C:/Users/test/kwangwoon/2024_1/산학연계/aws.ac.kwu.pem"

    pg_host = '127.0.0.1'
    pg_port = 5432
    pg_user = 'airflow'
    pg_password = 'airflow'
    pg_db = 'image_vector'

    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_private_key=ssh_private_key,
        remote_bind_address=(pg_host, pg_port),
        local_bind_address=('localhost', 5432)
    )
    tunnel.start()
    
    conn_pg = psycopg2.connect(
        host='localhost',
        port=tunnel.local_bind_port,
        user=pg_user,
        password=pg_password,
        dbname=pg_db
    )
    return conn_pg, tunnel

def translate_category_names(category_names):
    translator = Translator()
    translated_dict = {}
    for name in category_names:
        try:
            result = translator.translate(name, src='ko', dest='en')
            translated_dict[name] = result.text
        except Exception as e:
            print(f"Error translating {name}: {e}")
            translated_dict[name] = name
    return translated_dict

def mapping_translated_category(translated_dict):
    category_hierarchy = load_category_hierarchy()
    translated_category_hierarchy = {
        style_id: [[translated_dict.get(category, category) for category in sublist] for sublist in categories]
        for style_id, categories in category_hierarchy.items()
    }
    return translated_category_hierarchy

def load_cdn_urls():
    conn = get_pg_connection()[0]
    cursor = conn.cursor()
    sql_query = """
        SELECT DISTINCT cdn_url
        FROM image_vector
    """
    cursor.execute(sql_query)
    cdn_urls = {cdn_url for (cdn_url,) in cursor.fetchall()}
    conn.close()
    return cdn_urls


def process_categories(mall_type, categories):
    processed_categories = []
    for sublist in categories:
        if mall_type == 'musinsa':
            # musinsa일 때는 모든 요소를 ' '로 join
            category = ' '.join(sublist)
        elif mall_type in ['handsome', 'wconcept']:
            # handsome과 wconcept일 때는 처음 두 요소만 ' '로 join
            category = ' '.join(sublist[:2]) if len(sublist) >= 2 else ' '.join(sublist)
        processed_categories.append(category)
    
    return ', '.join(processed_categories)  # 모든 처리된 카테고리를 ', '로 join


def save_embeddings(mapped_dict):
    data_to_insert = []
    mall_type_mapping_dict = {'musinsa': "JN1qnDZA", 'wconcept': "l8WAu4fP", 'handsome': "FHyETFQN"}
    all_cdn_urls = fetch_cdn_urls()
    
    conn_pg, tunnel = get_pg_connection()
    conn_pg.autocommit = True
    cur = conn_pg.cursor()
    existing_cdn_urls = load_cdn_urls()

    try:
        for i, cdn_url in enumerate(all_cdn_urls):
            if cdn_url not in existing_cdn_urls:
                parts = cdn_url.split('/')
                style_id = parts[-2]
                mall_type_name = parts[-3]
                mall_type_id = mall_type_mapping_dict[mall_type_name]
                
                # mapping_dict에서 style_id에 해당하는 category를 찾기
                categories = mapped_dict.get(style_id, [])
               
                # category = ', '.join([' '.join(sublist) for sublist in categories])
                # mall type에 따라 categories 처리 방식 변경
                
                category = process_categories(mall_type_name, categories)
                
                try:
                    vec = process_image_and_feature(cdn_url, category)
                except Exception as e:
                    print(f'Error processing image: {e} - {cdn_url}')
                    print("An error occurred:")
                    traceback.print_exc()
                    continue
                data_to_insert.append((style_id, cdn_url, mall_type_id, vec))
                print(f'category : {category}, {i}번째 완료, url: {cdn_url}')

                if len(data_to_insert) >= 100:
                    psycopg2.extras.execute_batch(cur, """
                        INSERT INTO image_vector (style_id, cdn_url, mall_type_id, embedding) 
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (cdn_url) DO NOTHING
                    """, data_to_insert)
                    conn_pg.commit()
                    print('100개 아이템 데이터베이스에 삽입 완료')
                    data_to_insert = []

        if data_to_insert:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO image_vector (style_id, category, embedding, cdn_url) 
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (cdn_url) DO NOTHING
            """, data_to_insert)
            conn_pg.commit()
            print('잔여 데이터 데이터베이스에 삽입 완료')
    finally:
        cur.close()
        conn_pg.close()
        tunnel.close()

if __name__ == '__main__':
    category_names = load_category_names()
    print('완료')
    translated_dict = translate_category_name(category_names)
    print('완료2') 
    mapped_dict = mapping_translated_category(translated_dict)
    print('완료3')
    save_embeddings(mapped_dict)