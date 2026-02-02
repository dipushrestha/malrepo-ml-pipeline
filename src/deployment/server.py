from fastapi import FastAPI, UploadFile, File
import uvicorn
from src.deployment.inference import InferencePipeline

app = FastAPI(title="MLOps Model Server")
pipeline = InferencePipeline()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    prediction = pipeline.predict(image_bytes)
    return {"class_id": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
