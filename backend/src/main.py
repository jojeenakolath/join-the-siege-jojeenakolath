from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .services.classifier import DocumentClassifier
from .api.routes import files

app = FastAPI(
    title="Document Classification API",
    description="API for classifying various types of documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files.router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from pathlib import Path
# from fastapi import FastAPI, File, Form, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from .classifier import DocumentClassifier
# from .schema import DocumentResponse
# import uvicorn

# app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"])

# classifier = DocumentClassifier()

# @app.post("/classify", response_model=DocumentResponse)
# async def classify_document(file: UploadFile = File(...)):
#     try:
#         content = await file.read()
#         result = await classifier.classify_document(content, file.filename)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/classifydoc", response_model=DocumentResponse)
# async def classify_document(file: Path = Form(...)):
#     try:
#         content = file.read_bytes()
#         result = await classifier.classify_document(content, file)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/classify-batch")
# # async def classify_batch(files: List[UploadFile] = File(...)):
# #     try:
# #         file_data = [(await f.read(), f.filename) for f in files]
# #         return await classifier.process_batch(file_data)
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
    
# @app.get("/health")
# async def health_check():
#     """Health check endpoint for monitoring."""
#     return {"status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

