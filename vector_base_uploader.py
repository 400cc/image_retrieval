import boto3
import mysql.connector
from googletrans import Translator
import psycopg2
from typing import List
from psycopg2 import extras
import paramiko
from sshtunnel import SSHTunnelForwarder
import json
# from util.extract_image_feature import process_image_and_feature
from util.extract_image_feature import extractImageFeature
from util.mysql_db_util import get_db_connection, create_connection_pool
from load_category_hierarchy import get_category_hierarchy, load_category_hierarchy
import argparse
import gc
from util.pg_db_util import get_pg_connection
config_path = 'util/mysql_config.json'
connection_pool = create_connection_pool(config_path)

CATEGORY_MAPPING_FILE = "util/category_mapping.json"

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', type=str, default='cuda:0', help='# of gpu') 
    
    return parser

def load_category_mapping():
    with open(CATEGORY_MAPPING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

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

# def translate_category_name(category_names):
#     # 번역기 객체 생성
#     translator = Translator()

#     # 한글을 영어로 번역하여 딕셔너리 생성
#     translated_dict = {}
#     for name in category_names:
#         result = translator.translate(name, src='ko', dest='en')
#         translated_dict[name] = result.text
        
#     return translated_dict

def translate_category(category_name):
    translated_list = []
    translator = Translator()
    category_mapping_dict = load_category_mapping()
    for category in category_name:
        # 카테고리 매핑 딕셔너리에 있는 경우 매핑 사용
        if category in category_mapping_dict:
            translated_list.append(category_mapping_dict[category])
        else: 
            # 매핑이 없는 경우 Translator를 사용해 번역
            translated_category = translator.translate(category, src='ko', dest='en').text
            translated_list.append(translated_category)
     
    return translated_list

def mapping_translated_category(translated_dict):
    translator = Translator()
    
    category_hierarchy = load_category_hierarchy()
    
    translated_category_hierarchy = {}
    
    for style_id, categories in category_hierarchy.items():
        translated_category_hierarchy[style_id] = []
        
        for sublist in categories:
            translated_sublist = []
            
            for category in sublist:
                if category not in translated_dict:
                    try:
                        translated_category = translator.translate(category, src='ko', dest='en').text
                        translated_dict[category] = translated_category  # 번역 후 딕셔너리에 추가
                    except Exception as e:
                        print(f"Error translating {category}: {e}")
                        translated_category = category  # 오류 시 원래 값을 사용
                else:
                    translated_category = translated_dict[category]
                translated_sublist.append(translated_category)
            translated_category_hierarchy[style_id].append(translated_sublist)

    return translated_category_hierarchy


def fetch_cdn_urls(batch_size: int = 1000, last_offset: int = 0):
    conn = get_db_connection(connection_pool)
    cursor = conn.cursor()
    
    query = f"SELECT url FROM image LIMIT {batch_size} OFFSET {last_offset}"
    cursor.execute(query)
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    # URL 리스트와 다음 오프셋을 반환
    return [row[0] for row in rows], last_offset + len(rows)


# def translate_category_names(category_names):
#     translator = Translator()
#     translated_dict = {}
#     for name in category_names:
#         try:
#             result = translator.translate(name, src='ko', dest='en')
#             translated_dict[name] = result.text
#         except Exception as e:
#             print(f"Error translating {name}: {e}")
#             translated_dict[name] = name
#     return translated_dict

def mapping_translated_category(translated_dict):
    category_hierarchy = load_category_hierarchy()
    translated_category_hierarchy = {
        style_id: [[translated_dict.get(category, category) for category in sublist] for sublist in categories]
        for style_id, categories in category_hierarchy.items()
    }
    return translated_category_hierarchy

def load_cdn_urls(conn_pg):
    # conn = get_pg_connection()[0]
    cursor = conn_pg.cursor()
    sql_query = """
        SELECT DISTINCT cdn_url
        FROM image_vector
    """
    cursor.execute(sql_query)
    cdn_urls = {cdn_url for (cdn_url,) in cursor.fetchall()}
    # conn_pg.close()
    return cdn_urls


def process_categories(mall_type, categories):
    processed_categories = []
    for sublist in categories:
        if mall_type == 'musinsa':
            # musinsa일 때는 모든 요소를 ' '로 join
            category = ' '.join(sublist)
        elif mall_type in ['handsome', 'wconcept']:
            if 'Capsule Collection*' in sublist: 
                continue  # 캡슐컬렉션인 계층도는 빼버림
            # handsome과 wconcept일 때는 처음 두 요소만 ' '로 join
            category = ' '.join(sublist[:2]) if len(sublist) >= 2 else ' '.join(sublist)
        processed_categories.append(category)
        
    # 모든 처리된 카테고리를 ', '로 join
    final_category = ', '.join(processed_categories)
    
    if not final_category:  # final_category가 빈 문자열이면 'apparel'로 지정
        final_category = 'apparel'
        
    return final_category  


def save_embeddings(cdn_urls, mapped_dict, embedding):
    data_to_insert = []
    mall_type_mapping_dict = {'musinsa': "JN1qnDZA", 'wconcept': "l8WAu4fP", 'handsome': "FHyETFQN"}
    # all_cdn_urls = fetch_cdn_urls()
    # all_cdn_urls.reverse()
    
    conn_pg, tunnel = get_pg_connection()
    conn_pg.autocommit = True
    cur = conn_pg.cursor()
    existing_cdn_urls = load_cdn_urls(conn_pg)

    try:
        for i, cdn_url in enumerate(cdn_urls):
            if cdn_url not in existing_cdn_urls:
                parts = cdn_url.split('/')
                style_id = parts[-2]
                mall_type_name = parts[-3]
                
                if mall_type_name == 'wconcept_site':
                    mall_type_name = 'wconcept'
                
                mall_type_id = mall_type_mapping_dict[mall_type_name]
                
                # mapping_dict에서 style_id에 해당하는 category를 찾기
                categories = mapped_dict.get(style_id, [])
               
                # category = ', '.join([' '.join(sublist) for sublist in categories])
                # mall type에 따라 categories 처리 방식 변경
                
                category = process_categories(mall_type_name, categories)
                
                try:
                    vec = embedding.process_image_and_feature(cdn_url, category)
                except Exception as e:
                    print(f'Error processing image: {e} - {cdn_url}, {i} 번째, category: {category}')

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
                    gc.collect()

        if data_to_insert:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO image_vector (style_id, cdn_url, mall_type_id, embedding) 
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
    parser = argparse.ArgumentParser('Image Embedding', parents=[get_args_parser()])
    opts = parser.parse_args()
    device = opts.device
    embedding = extractImageFeature(device=device)
    category_mapping_dict = load_category_mapping()
    
    category_names = load_category_names()
    # print('category load')
    # translated_dict = translate_category_name(category_names)
    # print('category translation') 
    mapped_dict = mapping_translated_category(category_mapping_dict)
    print('category mapping')
    
    offset = 0
    batch_number = 0
    batch_size = 1000
    while True:
        cdn_urls, offset = fetch_cdn_urls(batch_size, offset)
        print(f"Processing batch {batch_number} with URLs starting from offset {offset - len(cdn_urls)}")
        batch_number += 1
        if not cdn_urls:
            break
        save_embeddings(cdn_urls, mapped_dict, embedding)
        
        del cdn_urls
        gc.collect()