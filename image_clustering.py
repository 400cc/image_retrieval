import logging
import json
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from util.pg_db_util import get_pg_connection

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 설정 로그
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def fetch_embedding_list(conn):
    query = """
    SELECT DISTINCT ON (style_id) style_id, embedding
    FROM image_vector
    ORDER BY style_id
    """
    df = pd.read_sql(query, conn)
    embeddings = df['embedding'].apply(json.loads)
    vectors = np.array(embeddings.tolist(), dtype=np.float32)
    style_ids = df['style_id'].tolist()
    return vectors, style_ids

def perform_clustering(vectors, n_clusters):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    clusters = kmeans.fit_predict(vectors)
    return clusters

def reduce_dimensions(vectors, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(vectors)

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("result.html", {"request": request, "image": None})

# TODO: request에 후보군까지 받도록 추가하기
@app.post("/clustering")
def cluster_and_visualize(request: Request, n_clusters: int = Form(...)):
    try:
        logger.info(f"Received request with n_clusters={n_clusters}")

        conn, tunnel = get_pg_connection()
        vectors, style_ids = fetch_embedding_list(conn)

        logger.info(f"Number of vectors fetched: {len(vectors)}")

        clusters = perform_clustering(vectors, n_clusters)
        vectors_2d = reduce_dimensions(vectors)
        
        data_points = [
            {
                "style_id": style_id,
                "x": float(vectors_2d[i, 0]),
                "y": float(vectors_2d[i, 1]),
                "cluster": int(clusters[i])
            }
            for i, style_id in enumerate(style_ids)
        ]
        
        return {"data_points": data_points}
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if conn:
            conn.close()
        if tunnel:
            tunnel.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)