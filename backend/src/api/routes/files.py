from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pathlib import Path
from ..schema import DocumentResponse
from ...services.classifier import DocumentClassifier

router = APIRouter(
    prefix="",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)

classifier = DocumentClassifier()

@router.post("/classify-webapp", response_model=DocumentResponse)
async def classify_document(
    file: UploadFile = File(...),
    classifier = classifier
):
    """
    Classify a document uploaded as a file.
    """
    try:
        content = await file.read()
        result = await classifier.classify_document(content, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify", response_model=DocumentResponse)
async def classify_document_from_path(
    file: Path = Form(...),
    classifier = classifier
):
    """
    Classify a document from a file path.
    """
    try:
        content = file.read_bytes()
        result = await classifier.classify_document(content, file.name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/batch", response_model=List[DocumentResponse])
# async def classify_batch(
#     files: List[UploadFile] = File(...),
#     classifier: DocumentClassifier = Depends(get_classifier)
# ):
#     """
#     Classify multiple documents in a batch.
#     """
#     try:
#         file_data = [(await f.read(), f.filename) for f in files]
#         return await classifier.process_batch(file_data)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

