from sqlalchemy import create_engine, Column, Text, String, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import logging

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
    
def find_similar_images(style_id_list, image_feature, offset = 5):
    session = SessionLocal()
    try:
        query = (
            session.query(
                ImageVector.cdn_url,
                ImageVector.style_id,
                (ImageVector.embedding.l2_distance(image_feature)).label('distance')
            )
            .filter(ImageVector.style_id.in_(style_id_list))
            .order_by('distance')
            .limit(offset)
        )
        logging.info(query)
        similar_images = query.all()
        style_image_dict = {}
        for similar_image in similar_images:
            style_id = similar_image.style_id
            cdn_url = similar_image.cdn_url
            if style_id not in style_image_dict:
                style_image_dict[style_id] = []
            style_image_dict[style_id].append(cdn_url)
        
    finally:
        session.close()
    
    return style_image_dict