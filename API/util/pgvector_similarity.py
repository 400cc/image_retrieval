from PIL import Image
import io
import base64
import psycopg2
from psycopg2.extras import execute_values
import logging

# PostgreSQL 연결 정보
DATABASE = {
    "dbname": "imagevector",
    "user": "test",
    "password": "5303",
    "host": "localhost"
}

def connect_to_db():
    conn = psycopg2.connect(**DATABASE)
    return conn

def find_similar_images(image_feature):
    conn = connect_to_db()
    cur = conn.cursor()
    
    # pg_similarity 모듈을 사용하여 유사한 이미지를 검색
    similarity_query = """
        SELECT id, embedding <-> %s::vector AS distance
        FROM items
        ORDER BY distance ASC
        LIMIT 5;
    """
    cur.execute(similarity_query, (image_feature,))
    similar_images = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return similar_images