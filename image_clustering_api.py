from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
from typing import List
import logging
from util.image_clustering import cluster_and_reduce

app = FastAPI()

# 설정 로그
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringRequest(BaseModel):
    style_id_list: List[str]
    n_clusters: int

@app.post("/clustering")
def process_clustering(request: ClusteringRequest):
    try:
        logger.info(f"Received clustering request: {request}")
        data_points = cluster_and_reduce(request.style_id_list, request.n_clusters)
        return {"data_points": data_points}
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
