from fastapi import FastAPI, Query, HTTPException
from typing import List
from pydantic import BaseModel
from util.extract_image_feature import process_image_and_feature
app = FastAPI()

class Image_url_category(BaseModel):
    image_url_list: List[str]
    category_list : List[str]
    
@app.post("/", status_code=200)
async def process_lists(request : Image_url_category):
    vectorized_result_list = []
    try:
        for index in range(len(request.image_url_list)):
            vectorized_result = process_image_and_feature(request.image_url_list[index], request.category_list[index])
            vectorized_result_list.append(vectorized_result)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))
    
    return {"vectorized_response" : vectorized_result_list}
