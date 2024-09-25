import logging
import json
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import umap
from typing import List
from util.pg_db_util import get_pg_connection

# 설정 로그
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_embedding_list(conn, mall_type_id: str, category_list: List[int]):

    query = """
    SELECT DISTINCT ON (style_id) i.style_id, i.embedding, i.cdn_url
    FROM image_vector i
    JOIN category_style c ON i.style_id = c.style_id
    """
    
    # category_list가 비어있지 않은 경우에만 조건 추가
    if category_list:
        query += "WHERE c.category_id IN %s\n"
        params = (tuple(category_list),)
    else:
        # category_list가 비어있는 경우 mall_type_id로 필터링
        query += "WHERE i.mall_type_id = %s\n"
        params = (mall_type_id,)
    
    df = pd.read_sql(query, conn, params=params)
    
    embeddings = df['embedding'].apply(json.loads)
    vectors = np.array(embeddings.tolist(), dtype=np.float32)
    
    style_ids = df['style_id'].tolist()
    urls = df['cdn_url'].tolist()
    
    return vectors, style_ids, urls

def perform_clustering(vectors: np.ndarray, n_clusters: int) -> np.ndarray:
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    clusters = kmeans.fit_predict(vectors)
    return clusters

def reduce_dimensions(vectors: np.ndarray, n_neighbors=10, min_dist=0.1, n_jobs=-1, learning_rate=1.0) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_jobs=n_jobs,
        learning_rate=learning_rate
    )
    return reducer.fit_transform(vectors)

def cluster_and_reduce(n_clusters: int, mall_type_id: str, category_list: List[int]):
    conn, tunnel = get_pg_connection()
    try:
        vectors, style_ids, urls = fetch_embedding_list(conn, mall_type_id, category_list)
        logger.info(f"Number of vectors fetched: {len(vectors)}")

        clusters = perform_clustering(vectors, n_clusters)
        vectors_2d = reduce_dimensions(vectors)
        
        data_points = [
            {
                "style_id": style_id,
                "x": float(vectors_2d[i, 0]),
                "y": float(vectors_2d[i, 1]),
                "cluster": int(clusters[i]),
                "url": urls[i]   
            }
            for i, style_id in enumerate(style_ids)
        ]
        
        return data_points

    finally:
        if conn:
            conn.close()
        if tunnel:
            tunnel.stop()
