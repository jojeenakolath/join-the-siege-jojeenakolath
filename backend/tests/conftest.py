import pytest
from PIL import Image, ImageDraw
import io
import os

@pytest.fixture(scope="session")
def test_data_dir():
    """Create and return test data directory."""
    dir_path = "tests/test_data"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

@pytest.fixture
def create_test_image():
    """Factory fixture to create test images with text."""
    def _create_image(text: str, size=(200, 200)):
        img = Image.new('RGB', size, color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 10), text, fill='black')
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
    
    return _create_image

@pytest.fixture
def mock_model_response():
    """Mock model response fixture."""
    return {
        'logits': [[0.1, 0.8, 0.1]],
        'hidden_states': None,
        'attentions': None
    }