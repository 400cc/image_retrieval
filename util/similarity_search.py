from sqlalchemy import create_engine, Column, Text, func
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

class ImageEmbedding(Base):
    __tablename__ = "version4"
    style_id = Column(Text)
    cdn_url = Column(Text, primary_key=True, index=True)
    category = Column(Text)
    embedding = Column(Vector(768))
    
def find_similar_images(image_feature, top_num = 5):
    session = SessionLocal()
    try:
        # query = (
        #     session.query(ImageEmbedding.cdn_url, (ImageEmbedding.embedding.l2_distance(image_feature)).label('distance'))
        #     .order_by('distance')
        #     .limit(top_num)
        # )

        subquery = (
            session.query(
                ImageEmbedding.style_id,
                func.min(ImageEmbedding.embedding.l2_distance(image_feature)).label('min_distance')
            )
            .group_by(ImageEmbedding.style_id)
            .subquery()
        )
        query = (
            session.query(
                ImageEmbedding.cdn_url,
                ImageEmbedding.style_id,
                (ImageEmbedding.embedding.l2_distance(image_feature)).label('distance')
            )
            .join(subquery, (ImageEmbedding.style_id == subquery.c.style_id) & (ImageEmbedding.embedding.l2_distance(image_feature) == subquery.c.min_distance))
            .order_by('distance')
            .limit(top_num)
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