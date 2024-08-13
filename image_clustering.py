import psycopg2
from sshtunnel import SSHTunnelForwarder
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from util.pg_db_util import get_pg_connection

app = FastAPI()

# 설정: Jinja2 템플릿 디렉토리
templates = Jinja2Templates(directory="templates")

# 선택적: 정적 파일 서빙 (예: CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

def fetch_embedding_list(conn):
    query = "SELECT embedding FROM image_vector"
    df = pd.read_sql(query, conn)
    return df['embedding'].tolist()

def perform_clustering(vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(vectors)

def reduce_dimensions(vectors, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(vectors)

def visualize_clusters(vectors_2d, clusters):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1], hue=clusters, palette="viridis")
    plt.title("2D Visualization of Clusters")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

class ClusterRequest(BaseModel):
    n_clusters: int

@app.post("/clustering")
def cluster_and_visualize(request: ClusterRequest, req: Request):
    try:
        conn, tunnel = get_pg_connection()
        vectors = fetch_embedding_list(conn)
        clusters = perform_clustering(vectors, request.n_clusters)
        vectors_2d = reduce_dimensions(vectors)
        img_str = visualize_clusters(vectors_2d, clusters)
        return templates.TemplateResponse("result.html", {"request": req, "image": img_str})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
        if tunnel:
            tunnel.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
