from sqlalchemy import create_engine, Column, Text, String, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import logging
from vector_base_uploader import get_pg_connection

# PostgreSQL 연결 정보
DATABASE = {
    "dbname": "imagevector",
    "user": "test",
    "password": "5303",
    "host": "localhost",
    "port": 5432
}

# SQLAlchemy 엔진 생성
DATABASE_URL = f"postgresql+psycopg2://{DATABASE['user']}:{DATABASE['password']}@{DATABASE['host']}:{DATABASE['port']}/{DATABASE['dbname']}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy 베이스 모델 생성
Base = declarative_base()

class ImageVector(Base):
    __tablename__ = "image_vector"
    style_id = Column(Text)
    cdn_url = Column(Text, primary_key=True, index=True)
    mall_type_id = Column(String(255))
    embedding = Column(Vector(768))

def build_filter(style_id_list, mall_type_id, image_feature, category, offset):
    # 기본 쿼리
    query = """
    SELECT
        cdn_url,
        style_id,
        mall_type_id,
        embedding <-> %s::vector AS distance
    FROM
        image_vector
    """
    
    conditions = []
    params = [image_feature]  # image_feature는 함수 외부에서 전달받는 변수라고 가정합니다.
    
    if mall_type_id is not None and category != "apparel":
        conditions.append("style_id IN %s")
        params.append(tuple(style_id_list))
    elif mall_type_id is not None and category == "apparel":
        conditions.append("mall_type_id = %s")
        params.append(mall_type_id)

    if conditions:
        query += "WHERE " + "AND ".join(conditions)

    query += """
    ORDER BY 
        distance
    """
    # 쿼리에서 LIMIT는 제거합니다.
    
    return query, params

def find_similar_images(style_id_list, mall_type_id, category, image_feature, offset=5):
    conn_pg, tunnel = get_pg_connection()
    try:
        cursor = conn_pg.cursor()
        query, params = build_filter(style_id_list, mall_type_id, image_feature, category, offset)
        
        logging.info("Executing query: %s with params: %s", query, params)
        cursor.execute(query, params)
        similar_images = cursor.fetchall()

        # 중복된 style_id 제거
        seen_style_ids = set()
        results = []
        for row in similar_images:
            cdn_url, style_id, mall_type_id, distance = row
            if style_id not in seen_style_ids:
                print(f'style_id: {style_id}')
                seen_style_ids.add(style_id)
                result = {
                    'cdn_url': cdn_url,
                    'style_id': style_id,
                    'mall_type_id': mall_type_id,
                    'distance': distance,
                }
                results.append(result)
            if len(results) >= offset:
                print(f'seen_style_ids: {seen_style_ids}')
                break
        
        cursor.close()
        return results

    finally:
        conn_pg.close()
        tunnel.close()
