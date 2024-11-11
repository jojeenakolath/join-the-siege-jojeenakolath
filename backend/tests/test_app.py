import pytest
import torch
from PIL import Image, ImageDraw
import io
from src.services.classifier import DocumentClassifier, DocumentType
from unittest.mock import patch

@pytest.fixture
def classifier():
    """Create a classifier instance for testing."""
    return DocumentClassifier()

@pytest.fixture
def sample_text():
    """Sample texts for different document types."""
    return {
        DocumentType.INVOICE: """
            INVOICE
            Invoice #: 12345
            Date: 2024-01-01
            Amount Due: $500.00
            Bill To: John Doe
            Items:
            1. Service A - $300.00
            2. Service B - $200.00
            Total: $500.00
        """,
        DocumentType.BANK_STATEMENT: """
            BANK STATEMENT
            Account: 1234567890
            Period: January 2024
            Opening Balance: $1,000.00
            Transactions:
            - Deposit: $500.00
            - Withdrawal: $200.00
            Closing Balance: $1,300.00
        """
    }

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

def test_classifier_initialization(classifier):
    """Test classifier initialization."""
    assert classifier.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert hasattr(classifier, "models")
    assert isinstance(classifier.models, dict)

@pytest.mark.asyncio
async def test_extract_text(classifier, sample_image):
    """Test text extraction from image."""
    with patch('pytesseract.image_to_string', return_value="INVOICE #12345"):
        text, _ = await classifier._extract_text(sample_image, ".png")
        assert text.strip() == "INVOICE #12345"

def test_get_rule_based_scores(classifier, sample_text):
    """Test rule-based classification scoring."""
    # Note: Changed to match your implementation
    text = sample_text[DocumentType.INVOICE]
    scores = classifier._get_rule_based_scores(text)
    assert scores[DocumentType.INVOICE] > scores[DocumentType.BANK_STATEMENT]
    assert all(isinstance(score, float) for score in scores.values())

@pytest.mark.asyncio
async def test_error_handling_invalid_image(classifier):
    """Test error handling with invalid image data."""
    invalid_content = b"not an image"
    result = await classifier._extract_text(invalid_content, ".png")
    assert result[0] == ""  # Should return empty string for text
    assert isinstance(result[1], dict)  # Should return dict for metadata

@pytest.mark.asyncio
async def test_error_handling_invalid_extension(classifier):
    """Test error handling with invalid file extension."""
    with pytest.raises(ValueError) as exc_info:
        await classifier.classify_document(b"some content", "test.xyz")
    assert "Unsupported file type" in str(exc_info.value)

@pytest.mark.asyncio
async def test_error_handling_empty_content(classifier):
    """Test error handling with empty content."""
    empty_content = b""
    result = await classifier._extract_text(empty_content, ".pdf")
    assert result[0] == ""
    assert isinstance(result[1], dict)

@pytest.mark.asyncio
async def test_error_handling_corrupted_pdf(classifier):
    """Test error handling with corrupted PDF."""
    corrupted_pdf = b"%PDF-1.4\ncorrupted content"
    result = await classifier._extract_text(corrupted_pdf, ".pdf")
    assert result[0] == ""
    assert isinstance(result[1], dict)

@pytest.mark.asyncio
async def test_error_handling_classification(classifier):
    """Test error handling during classification process."""
    # Create an image that will cause OCR to fail
    img = Image.new('RGB', (1, 1), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    result = await classifier.classify_document(img_byte_arr.getvalue(), "test.png")
    assert result.document_type == DocumentType.UNKNOWN
    assert result.confidence_score == 0.0

@pytest.mark.asyncio
async def test_error_handling_invalid_extension(classifier):
    """Test error handling with invalid file extension."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        await classifier.classify_document(b"content", "test.xyz")

@pytest.mark.asyncio
async def test_error_handling_ml_model(classifier):
    """Test error handling when ML model fails."""
    content = b"test content"
    with patch.object(classifier, '_get_ml_classification', side_effect=Exception("Model error")):
        result = await classifier.classify_document(content, "test.pdf")
        assert result.document_type == DocumentType.UNKNOWN
        assert result.confidence_score == 0.0
        assert "classification_error" in result.metadata
        assert "Model error" in result.metadata["classification_error"]

@pytest.mark.asyncio
async def test_error_handling_file_size(classifier):
    """Test error handling with large file."""
    # Create a file larger than MAX_FILE_SIZE
    large_content = b"x" * (classifier.MAX_FILE_SIZE + 1)
    
    result = await classifier.classify_document(large_content, "large.png")
    assert result.document_type == DocumentType.UNKNOWN
    assert "file_size" in result.metadata
    assert "error" in result.metadata
    assert "File too large" in result.metadata["error"]

@pytest.mark.asyncio
async def test_error_handling_ocr(classifier):
    """Test error handling when OCR fails."""
    with patch('pytesseract.image_to_string', side_effect=Exception("OCR error")):
        result = await classifier._extract_text(b"some content", ".png")
        assert result[0] == ""
        assert isinstance(result[1], dict)

def test_model_loading():
    """Test model loading with error handling."""
    with patch('transformers.DistilBertTokenizer.from_pretrained', side_effect=Exception):
        classifier = DocumentClassifier()
        assert classifier.models['text']['enabled'] is False

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_classification_pipeline(classifier):
    """Integration test for full classification pipeline."""
    # Create a test document with known content
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "INVOICE\nInvoice #: 12345\nAmount: $500", fill='black')
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Mock the text extraction and classification
    with patch.object(classifier, '_extract_text', 
                     return_value=("INVOICE\nInvoice #: 12345\nAmount: $500", {})), \
         patch.object(classifier, '_get_ml_classification',
                     return_value=(DocumentType.INVOICE, 0.9)):
        
        result = await classifier.classify_document(img_byte_arr.getvalue(), "test.png")
        assert result.document_type == DocumentType.INVOICE
        assert result.confidence_score > 0
        assert isinstance(result.possible_types, list)

@pytest.mark.asyncio
async def test_model_inference(classifier):
    """Test model inference."""
    if classifier.models['text']['enabled']:
        text = "INVOICE #12345 Amount Due: $500.00"
        tokenizer = classifier.models['text']['tokenizer']
        model = classifier.models['text']['model']
        
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(classifier.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            assert outputs.logits.shape[1] == len(DocumentType)

@pytest.mark.asyncio
async def test_classify_document_with_high_confidence(classifier, sample_image):
    """Test document classification with high confidence."""
    with patch.object(classifier, '_extract_text',
                     return_value=("INVOICE #12345", {})), \
         patch.object(classifier, '_get_ml_classification',
                     return_value=(DocumentType.INVOICE, 0.95)):
        
        result = await classifier.classify_document(sample_image, "test.png")
        assert result.document_type == DocumentType.INVOICE
        assert result.confidence_score >= 0.9
        assert isinstance(result.metadata, dict)

@pytest.mark.asyncio
async def test_classify_document_unknown(classifier, sample_image):
    """Test classification of unknown document type."""
    with patch.object(classifier, '_extract_text',
                     return_value=("Random text", {})), \
         patch.object(classifier, '_get_ml_classification',
                     return_value=(DocumentType.UNKNOWN, 0.1)):
        
        result = await classifier.classify_document(sample_image, "test.png")
        assert result.document_type == DocumentType.UNKNOWN
        assert result.confidence_score < 0.5

@pytest.mark.parametrize("doc_type,text", [
    (DocumentType.INVOICE, "INVOICE #12345"),
    (DocumentType.BANK_STATEMENT, "BANK STATEMENT"),
    (DocumentType.PAYSLIP, "PAYSLIP DETAILS"),
])
@pytest.mark.asyncio
async def test_different_document_types(classifier, sample_image, doc_type, text):
    """Test classification of different document types."""
    with patch.object(classifier, '_extract_text',
                     return_value=(text, {})), \
         patch.object(classifier, '_get_ml_classification',
                     return_value=(doc_type, 0.9)):
        
        result = await classifier.classify_document(sample_image, "test.png")
        assert result.document_type == doc_type

@pytest.mark.asyncio
async def test_metadata_generation(classifier, sample_image):
    """Test metadata generation in classification."""
    with patch.object(classifier, '_extract_text',
                     return_value=("Test content", {"width": 100, "height": 100})), \
         patch.object(classifier, '_get_ml_classification',
                     return_value=(DocumentType.INVOICE, 0.8)):
        
        result = await classifier.classify_document(sample_image, "test.png")
        assert "filename" in result.metadata
        assert "file_size" in result.metadata