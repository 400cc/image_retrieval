from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from util.pg_db_util import get_pg_connection

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("result.html", {"request": request, "image": None})

@app.post("/clustering")
def cluster_and_visualize(request: Request, n_clusters: int):
    try:
        if n_clusters < 3 or n_clusters % 2 == 0:
            raise HTTPException(status_code=400, detail="Number of clusters must be an odd integer greater than or equal to 3.")
        
        conn, tunnel = get_pg_connection()
        vectors = fetch_embedding_list(conn)
        clusters = perform_clustering(vectors, n_clusters)
        vectors_2d = reduce_dimensions(vectors)
        img_str = visualize_clusters(vectors_2d, clusters)
        return templates.TemplateResponse("result.html", {"request": request, "image": img_str})
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
