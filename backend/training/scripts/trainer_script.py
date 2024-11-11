from typing import Dict, List, Optional, Union
import pdf2image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentType(str, Enum):
    DRIVERS_LICENSE = "drivers_license"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"
    PAYSLIP = "payslip"
    TAX_RETURN = "tax_return"
    UTILITY_BILL = "utility_bill"
    UNKNOWN = "unknown"

class DocumentDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 512,
        transform=None
    ):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        
        # Create label mapping
        self.label2id = {label: idx for idx, label in enumerate(DocumentType)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item['text']
        label = self.label2id[item['label']]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DocumentTrainer:
    def __init__(
        self,
        model_name: str = "roberta-base",
        output_dir: str = "src/trained_models",
        batch_size: int = 16,
        num_epochs: int = 5,
        learning_rate: float = 2e-5
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(DocumentType),
            id2label={i: label.value for i, label in enumerate(DocumentType)},
            label2id={label.value: i for i, label in enumerate(DocumentType)}
        )

    async def prepare_training_data(
        self,
        raw_data_dir: Path,
        output_path: Path
    ) -> Path:
        """Prepare training data from raw documents."""
        data = []
        
        # Process each document type
        for doc_type in DocumentType:
            if doc_type == DocumentType.UNKNOWN:
                continue
                
            doc_dir = raw_data_dir / doc_type.value
            if not doc_dir.exists():
                logger.warning(f"Directory not found for {doc_type.value}")
                continue
            
            # Process each document
            files = list(doc_dir.glob("*"))
            if not files:
                logger.warning(f"No files found for {doc_type.value}")
                continue
                
            for file_path in tqdm(files, desc=f"Processing {doc_type.value}"):
                try:
                    # Extract text from document
                    text = self._extract_text(file_path)
                    
                    if text:
                        # Add to dataset
                        data.append({
                            'text': text,
                            'label': doc_type.value,
                            'source': file_path.name
                        })
                        
                        # Add augmented versions
                        augmented_texts = self._augment_text(text)
                        for aug_text in augmented_texts:
                            data.append({
                                'text': aug_text,
                                'label': doc_type.value,
                                'source': f"aug_{file_path.name}"
                            })
                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        if not data:
            raise ValueError("No training data was generated!")
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} training examples to {output_path}")
        
        return output_path

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from document file."""
        try:
            if file_path.suffix.lower() == '.pdf':
                # Convert PDF to images
                images = pdf2image.convert_from_path(str(file_path))
                text = ""
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"
                return text
            
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Process image directly
                image = Image.open(file_path)
                return pytesseract.image_to_string(image)
            
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""

    def _augment_text(self, text: str) -> List[str]:
        """Create augmented versions of text."""
        augmented = []
        
        # Add random noise
        noisy = self._add_noise(text)
        if noisy != text:
            augmented.append(noisy)
        
        # Remove random words
        dropped = self._drop_words(text)
        if dropped != text:
            augmented.append(dropped)
        
        return augmented

    def _add_noise(self, text: str, noise_level: float = 0.1) -> str:
        """Add random noise to text."""
        words = text.split()
        num_words = len(words)
        num_noise = int(num_words * noise_level)
        
        for _ in range(num_noise):
            idx = np.random.randint(0, num_words)
            words[idx] = words[idx][::-1]  # Reverse the word as noise
        
        return " ".join(words)

    def _drop_words(self, text: str, drop_rate: float = 0.1) -> str:
        """Randomly drop words from text."""
        words = text.split()
        mask = np.random.rand(len(words)) > drop_rate
        return " ".join(np.array(words)[mask])

    def train(
        self,
        train_path: Path,
        val_path: Optional[Path] = None,
        experiment_name: str = "document_classifier"
    ) -> Dict:
        """Train the model."""
        # Create datasets
        train_dataset = DocumentDataset(train_path, self.tokenizer)
        val_dataset = DocumentDataset(val_path, self.tokenizer) if val_path else None
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train model
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model(str(self.output_dir / "final"))
        
        return train_result.metrics
