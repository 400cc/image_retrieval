from sqlalchemy import create_engine, Column, Text, String, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import logging
from util.pg_db_util import get_pg_connection


# SQLAlchemy 베이스 모델 생성
Base = declarative_base()

class ImageVector(Base):
    __tablename__ = "image_vector"
    style_id = Column(Text)
    cdn_url = Column(Text, primary_key=True, index=True)
    mall_type_id = Column(String(255))
    embedding = Column(Vector(768))

def build_filter(mall_type_id, image_feature, category_id_list):
    query = """
    SELECT
        i.cdn_url,
        i.style_id,
        i.mall_type_id,
        i.embedding <=> %s::vector AS distance
    FROM
        image_vector i
    """
    
    conditions = []
    params = [image_feature]
    
    if mall_type_id is not None and category_id_list:
        query += """
        JOIN category_style c ON i.style_id = c.style_id
        """
        conditions.append("c.category_id IN %s")
        params.append(tuple(category_id_list))
    elif mall_type_id is not None and not category_id_list:
        conditions.append("i.mall_type_id = %s") 
        params.append(mall_type_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += """
    ORDER BY distance
    LIMIT 100
    """
    # 쿼리에서 LIMIT는 제거합니다.
    
    return query, params

def find_similar_images(mall_type_id, category_name, category_id_list, image_feature, offset=5):
    conn_pg, tunnel = get_pg_connection()
    try:
        cursor = conn_pg.cursor()
        query, params = build_filter(mall_type_id, image_feature, category_id_list)
        logging.info("category : %s", category_name)
        logging.info("Executing query: %s with params: %s", query, params)
        cursor.execute(query, params)
        similar_images = cursor.fetchall()
        # 중복된 style_id 제거
        seen_style_ids = set()
        results = []
        for row in similar_images:
            cdn_url, style_id, mall_type_id, distance = row
            if style_id not in seen_style_ids:
                seen_style_ids.add(style_id)
                result = {
                    'cdn_url': cdn_url,
                    'style_id': style_id,
                    'mall_type_id': mall_type_id,
                    'distance': distance,
                }
                results.append(result)
            if len(results) >= offset:
                break
        
        cursor.close()
        return results
    
    finally:
        conn_pg.close()
        tunnel.close()
