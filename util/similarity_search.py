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
    
def find_similar_images(style_id_list, image_feature, offset=5):
    conn_pg, tunnel = get_pg_connection()
    try:
        cursor = conn_pg.cursor()

        query = """
        SELECT
            cdn_url,
            style_id,
            mall_type_id,
            embedding <-> %s AS distance
        FROM
            image_vector
        WHERE
            style_id = ANY(%s)
        ORDER BY
            distance
        LIMIT %s;
        """

        cursor.execute(query, (image_feature, style_id_list, offset))
        similar_images = cursor.fetchall()

        results = []
        for row in similar_images:
            cdn_url, style_id, mall_type_id, distance = row
            result = {
                'cdn_url': cdn_url,
                'style_id': style_id,
                'mall_type_id': mall_type_id,
            }
            results.append(result)
        
        cursor.close()
        return results

    finally:
        conn_pg.close()
        tunnel.close()
    