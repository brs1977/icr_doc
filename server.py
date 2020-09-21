import cv2
import os
import io
import numpy as np
from typing import List
from doc_extractor import extract, read_image

from fastapi import Request, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):    
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.COLOR_BGR2RGB) 
    json = extract(image)

    return JSONResponse(content=json)
