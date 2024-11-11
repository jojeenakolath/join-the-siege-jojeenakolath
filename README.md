# Heron Coding Challenge - File Classifier

## Features

- Multi-format support 
- High-accuracy classification using hybrid approach (ML + Rule-based)
- User-friendly web interface
- RESTful API endpoints
- Support for batch processing
- Automatic OCR text extraction

## Setup

```
project/
├── backend/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   └── files.py
│   │   │   └── schema.py
│   │   └── services/
│   │       ├── __init__.py
│   │       └── classifier.py
│   ├── models/
│   │   └── trained_model/
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_app.py
│   └── requirements.txt
├── frontend/
    ├── streamlit_app.py
    └── requirements.txt

```

## Installation

### Prerequisites

```bash
# System dependencies
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# For MacOS
brew install tesseract poppler

# For Windows
# Download and install Tesseract and Poppler manually
```

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
pip install -r requirements.txt
```

## Running the Application

1. Start the Backend:

```bash
cd backend
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

2. Start the Frontend:

```bash
cd frontend
streamlit run streamlit_app.py --server.enableXsrfProtection=false
```

The application will be available at:

- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Using the API

### Classify a Document (File Upload)

```bash
curl -X POST "http://localhost:8000/classify-webapp" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

### Classify from Path

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "accept: application/json" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "file=/absolute/path/to/your/document.pdf"
```

### Python Example

```python
import requests

# Upload file
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        files={'file': f}
    )
print(response.json())
```

## Solution Overview

### 1. Document Input Validation

- **Supported Formats**: PDF, PNG, JPG, JPEG
- **Size Limit**: 10MB
- **Validation Checks**:
  - File extension verification
  - File size validation
  - Content integrity check

### 2. Text Extraction Pipeline

#### 2.1 PDF Processing

- Convert PDF to images using `pdf2image`
- Process each page individually
- Combine extracted text from all pages
- Store page count in metadata

#### 2.2 Image Processing

- Load image using PIL
- Apply image enhancement:
  - Denoise
  - Adjust contrast
  - Optimize for OCR
- Extract text using Tesseract OCR
- Store image metadata (size, mode)

### 3. Classification Engine

The system employs a dual-classification approach:

#### 3.1 Machine Learning Classification

- **Model**: DistilBERT-based classifier
- **Process**:
  - Tokenize extracted text
  - Generate embeddings
  - Predict document type
  - Calculate confidence score
- **Confidence Threshold**: 0.7 for high confidence

#### 3.2 Rule-Based Classification

- **Keyword Hierarchy**:
  - High importance (weight: 1.0)
  - Medium importance (weight: 0.6)
  - Low importance (weight: 0.3)
- **Scoring**:
  - Exact matches
  - Partial matches (0.5 weight)
  - Normalized scores

### 4. Result Determination

The system uses the following logic to determine the final classification:

```python
if ml_confidence > 0.7:
    # Use ML classification
    final_type = ml_prediction
    confidence = ml_confidence
else:
    # Compare ML and rule-based results
    if rule_based_score > ml_confidence:
        final_type = rule_based_prediction
        confidence = rule_based_score
    else:
        final_type = ml_prediction
        confidence = ml_confidence
```

### 5. Response Format

```json
{
    "document_type": "string",
    "confidence_score": "float",
    "possible_types": [
        ["type", "score"],
        ...
    ],
    "metadata": {
        "filename": "string",
        "file_size": "int",
        "processing_time": "float",
        "text_length": "int",
        ...
    }
}
```

## Synthetic Dataset Preparation

Since real-world document data can be difficult to obtain, the code generates a small synthetic dataset to quickly demonstrate the training process. The prepare_data() function creates a dataset with the following characteristics:

6 document types are defined: DocumentType.DRIVERS_LICENSE, DocumentType.BANK_STATEMENT, DocumentType.INVOICE, DocumentType.PAYSLIP, DocumentType.TAX_RETURN, DocumentType.UTILITY_BILL
For each document type, 3 sample text snippets are provided, representing the kind of content that might be found in those documents.
To add some variation, 2 additional samples are created for each original text by randomly shuffling the words.
The dataset is then split into training and validation sets using train_test_split().

### Model Training

The DocumentTrainer class handles the model training process:

It initializes the DistilBERT tokenizer and sequence classification model, setting the device to use either the GPU (if available) or the CPU.
The training hyperparameters are set, including the maximum sequence length (128), batch size (16), and number of epochs (2).
The train() function:

Calls prepare_data() to load the training and validation datasets.
Creates PyTorch DataLoader objects for the datasets.
Defines an AdamW optimizer with a learning rate of 5e-5.
Runs the training loop for the specified number of epochs, tracking the training loss and validation accuracy.
Saves the trained model and tokenizer to the trained_models directory.

There are two training scripts : trainer_script.py uses MLFlow, Weights and Biases for ML training and experimenting, however the process might take too long hence have added a faster script, trasiner_script_fast.py

## Test Suite

### Fixtures

classifier(): Creates an instance of the DocumentClassifier for testing.
sample_text(): Provides sample text content for different document types.
sample_image(): Generates a sample image for testing.

### Test Cases

Initialization, text extraction, and rule-based scoring.
Error handling for invalid inputs, model failures, and large files.
Integration test for the full classification pipeline.
Validation of model inference capabilities.
Testing document classification with varying confidence levels and types.
Checking metadata generation during classification.

The test suite ensures the reliability and robustness of the DocumentClassifier class by covering a wide range of scenarios, from happy paths to edge cases and error handling.

## CI/CD

A sample CI/CD workflow template has been added for potentially dockerising and deploying to AWS

## Health Checks

```bash
curl http://localhost:8000/health
```

## Some next Steps / Improvements

- Build Docker image of document classifier and deploy to cloud / registry
- Use Redis for caching
- Use MLFlow, Weights and Biases for better model training and experimenting
