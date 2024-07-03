from sqlalchemy import create_engine, Column, Text
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
    style_ID = Column(Text)
    cdn_url = Column(Text, primary_key=True, index=True)
    category = Column(Text)
    embedding = Column(Vector(768))
    
def find_similar_images(image_feature, top_num = 5):
    session = SessionLocal()
    try:
        query = (
            session.query(ImageEmbedding.cdn_url, (ImageEmbedding.embedding.l2_distance(image_feature)).label('distance'))
            .order_by('distance')
            .limit(top_num)
        )
        logging.info(query)
        similar_images = query.all()
        cdn_url_list = []
        for similar_image in similar_images :
            cdn_url_list.append(similar_image[0])
        
    finally:
        session.close()
    
    return cdn_url_list