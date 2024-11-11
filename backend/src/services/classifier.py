from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import pytesseract
from io import BytesIO
import pdf2image
import time

from ..api.schema import DocumentType, DocumentResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentClassifier:
    SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()
        logger.info(f"Classifier initialized using device: {self.device}")

    def _initialize_models(self):
        try:
            # self.models = {
            #     'text': {
            #         'tokenizer': DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
            #         'model': DistilBertForSequenceClassification.from_pretrained(
            #             'distilbert-base-uncased',
            #             num_labels=len(DocumentType)
            #         ).to(self.device),
            #         'enabled': True
            #     }
            # }
            self.models = {
                'text': {
                    'tokenizer': DistilBertTokenizer.from_pretrained('app/model_fast'),
                    'model': DistilBertForSequenceClassification.from_pretrained('app/model_fast'),
                    'enabled': True
                }
            }
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.models = {'text': {'enabled': False}}

    async def _extract_text(self, content: bytes, file_extension: str) -> Tuple[str, Dict]:
        """Extract text from document with improved error handling."""
        try:
            metadata = {
                "file_size": len(content),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            if len(content) > self.MAX_FILE_SIZE:
                logger.warning("File exceeds size limit")
                return "", {**metadata, "error": "File too large"}

            if file_extension == '.pdf':
                try:
                    images = pdf2image.convert_from_bytes(content)
                    text = ""
                    for img in images:
                        text += pytesseract.image_to_string(img) + "\n"
                    metadata["page_count"] = len(images)
                    return text, metadata
                except Exception as e:
                    logger.error(f"PDF processing failed: {e}")
                    return "", {**metadata, "error": str(e)}
            else:
                try:
                    image = Image.open(BytesIO(content))
                    metadata.update({
                        "image_size": image.size,
                        "image_mode": image.mode
                    })
                    text = pytesseract.image_to_string(image)
                    return text, metadata
                except Exception as e:
                    logger.error(f"Image processing failed: {e}")
                    return "", {**metadata, "error": str(e)}

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return "", {"error": str(e)}

    async def _get_ml_classification(self, text: str) -> Tuple[DocumentType, float]:
        """Get ML model classification with error handling."""
        try:
            if not self.models['text']['enabled'] or not text.strip():
                return DocumentType.UNKNOWN, 0.0

            inputs = self.models['text']['tokenizer'](
                text,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.models['text']['model'](**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)
                return list(DocumentType)[predicted_class.item()], confidence.item()

        except Exception as e:
            logger.error(f"ML classification failed: {e}")
            raise
    def _get_rule_based_scores(self, text: str) -> Dict[DocumentType, float]:
        """Calculate rule-based classification scores."""
        text = text.lower()
        scores = {doc_type: 0.0 for doc_type in DocumentType}
        
        # Define keywords for each document type
        keywords = {
            DocumentType.DRIVERS_LICENSE: {
                'high': ["driver license", "driver's license", "driving licence", "dl number"],
                'medium': ["driver", "license", "permit", "identification"],
                'low': ["vehicle", "class", "restrictions"]
            },
            DocumentType.BANK_STATEMENT: {
                'high': ["bank statement", "account statement", "balance summary"],
                'medium': ["account", "balance", "transaction", "deposit"],
                'low': ["credit", "debit", "transfer"]
            },
            DocumentType.INVOICE: {
                'high': ["invoice number", "invoice #", "bill to", "invoice total"],
                'medium': ["invoice", "total amount", "due date", "payment"],
                'low': ["subtotal", "tax", "amount"]
            },
            DocumentType.PAYSLIP: {
                'high': ["pay slip", "payslip", "salary statement", "wage statement"],
                'medium': ["salary", "wages", "earnings", "deductions"],
                'low': ["tax", "net", "gross"]
            },
            DocumentType.TAX_RETURN: {
                'high': ["tax return", "form 1040", "tax statement"],
                'medium': ["taxable income", "deductions", "tax year"],
                'low': ["refund", "credit", "filing"]
            },
            DocumentType.UTILITY_BILL: {
                'high': ["utility bill", "electric bill", "gas bill", "water bill"],
                'medium': ["meter reading", "usage", "service period"],
                'low': ["consumption", "rate", "charges"]
            }
        }
        
        # Weights for different keyword importance levels
        weights = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }
        
        # Calculate scores
        for doc_type, keyword_groups in keywords.items():
            doc_score = 0.0
            total_keywords = 0
            
            for importance, kw_list in keyword_groups.items():
                weight = weights[importance]
                total_keywords += len(kw_list)
                
                # Check for exact matches
                matches = sum(1 for kw in kw_list if kw in text)
                doc_score += matches * weight
                
                # Check for partial matches
                partial_matches = sum(
                    1 for kw in kw_list 
                    if any(word in text.split() for word in kw.split())
                    and kw not in text  # Don't count if already counted as exact match
                )
                doc_score += partial_matches * weight * 0.5
            
            # Normalize score
            if total_keywords > 0:
                scores[doc_type] = min(1.0, doc_score / (total_keywords * weights['high']))
        
        # Set unknown type score
        scores[DocumentType.UNKNOWN] = 0.1 if max(scores.values()) < 0.3 else 0.0
        
        return scores

    # async def classify_document(self, content: bytes, filename: str) -> DocumentResponse:
    #     """Classify document with comprehensive error handling."""
    #     start_time = time.time()
    #     metadata = {
    #         "filename": filename,
    #         "file_size": len(content),
    #         "processing_started": time.strftime("%Y-%m-%d %H:%M:%S")
    #     }

    #     try:
    #         # Validate file extension
    #         file_extension = Path(filename).suffix.lower()
    #         if file_extension not in self.SUPPORTED_EXTENSIONS:
    #             raise ValueError(f"Unsupported file type: {file_extension}")

    #         # Check file size
    #         if len(content) > self.MAX_FILE_SIZE:
    #             return DocumentResponse(
    #                 document_type=DocumentType.UNKNOWN,
    #                 confidence_score=0.0,
    #                 possible_types=[],
    #                 metadata={**metadata, "error": "File too large"}
    #             )

    #         # Extract text
    #         text, extract_metadata = await self._extract_text(content, file_extension)
    #         metadata.update(extract_metadata)

    #         try:
    #             # Get ML classification
    #             doc_type, ml_confidence = await self._get_ml_classification(text)
                
    #             # Get rule-based scores
    #             rule_scores = self._get_rule_based_scores(text)
                
    #             # Combine results
    #             if ml_confidence > 0.7:  # High ML confidence
    #                 final_type = doc_type
    #                 confidence = ml_confidence
    #             else:  # Use rule-based
    #                 max_score_type = max(rule_scores.items(), key=lambda x: x[1])[0]
    #                 final_type = max_score_type
    #                 confidence = rule_scores[max_score_type]

    #             # Get possible types
    #             possible_types = [
    #                 (t, s) for t, s in rule_scores.items()
    #                 if s > 0.1 and t != DocumentType.UNKNOWN
    #             ]
    #             possible_types.sort(key=lambda x: x[1], reverse=True)
                
    #             # If no good matches or confidence too low, mark as unknown
    #             if confidence < 0.3:
    #                 final_type = DocumentType.UNKNOWN
    #                 confidence = 0.0  # Set to 0.0 for unknown type

    #         except Exception as e:
    #             logger.error(f"Classification failed: {e}")
    #             metadata["classification_error"] = str(e)  
    #             return DocumentResponse(
    #                 document_type=DocumentType.UNKNOWN,
    #                 confidence_score=0.0,
    #                 possible_types=[],
    #                 metadata=metadata
    #             )

    #         # Calculate processing time
    #         processing_time = time.time() - start_time
    #         metadata["processing_time"] = processing_time

    #         return DocumentResponse(
    #             document_type=final_type,
    #             confidence_score=confidence,
    #             possible_types=possible_types,
    #             metadata=metadata
    #         )

    #     except ValueError as e:
    #         # Handle known validation errors
    #         logger.warning(f"Validation error: {e}")
    #         raise

    #     except Exception as e:
    #         # Handle unexpected errors
    #         logger.error(f"Unexpected error: {e}")
    #         return DocumentResponse(
    #             document_type=DocumentType.UNKNOWN,
    #             confidence_score=0.0,  # 0.0 confidence for errors
    #             possible_types=[],
    #             metadata={**metadata, "error": str(e)}
    #         )
    async def classify_document(self, content: bytes, filename: str) -> DocumentResponse:
        """Classify document with comprehensive error handling."""
        start_time = time.time()
        metadata = {
            "filename": filename,
            "file_size": len(content),
            "processing_started": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            # Validate file extension
            file_extension = Path(filename).suffix.lower()
            if file_extension not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Check file size
            if len(content) > self.MAX_FILE_SIZE:
                return DocumentResponse(
                    document_type=DocumentType.UNKNOWN,
                    confidence_score=0.0,
                    possible_types=[],
                    metadata={**metadata, "error": "File too large"}
                )

            # Extract text
            text, extract_metadata = await self._extract_text(content, file_extension)
            metadata.update(extract_metadata)

            try:
                # Get ML classification
                doc_type, ml_confidence = await self._get_ml_classification(text)
                
                # Get rule-based scores
                rule_scores = self._get_rule_based_scores(text)
                
                # Get possible types with scores
                possible_types = [
                    (t, s) for t, s in rule_scores.items()
                    if s > 0.0 and t != DocumentType.UNKNOWN  # Include all non-zero scores
                ]
                possible_types.sort(key=lambda x: x[1], reverse=True)

                # Combine results
                if ml_confidence > 0.7:  # High ML confidence
                    final_type = doc_type
                    confidence = ml_confidence
                else:  # Use rule-based or best possible type
                    # Get best type from either ML or rule-based
                    best_rule_type, best_rule_score = max(rule_scores.items(), key=lambda x: x[1])
                    
                    if best_rule_score > ml_confidence:
                        final_type = best_rule_type
                        confidence = best_rule_score
                    else:
                        final_type = doc_type
                        confidence = ml_confidence

                # Only set as UNKNOWN if no confidence at all
                if confidence == 0.0 and not possible_types:
                    final_type = DocumentType.UNKNOWN
                elif confidence == 0.0 and possible_types:
                    # Use the best possible type if available
                    final_type, confidence = possible_types[0]

            except Exception as e:
                logger.error(f"Classification failed: {e}")
                metadata["classification_error"] = str(e)  
                return DocumentResponse(
                    document_type=DocumentType.UNKNOWN,
                    confidence_score=0.0,
                    possible_types=[],
                    metadata=metadata
                )

            # Calculate processing time
            processing_time = time.time() - start_time
            metadata["processing_time"] = processing_time

            return DocumentResponse(
                document_type=final_type,
                confidence_score=confidence,
                possible_types=possible_types,
                metadata=metadata
            )

        except ValueError as e:
            # Handle known validation errors
            logger.warning(f"Validation error: {e}")
            raise

        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error: {e}")
            return DocumentResponse(
                document_type=DocumentType.UNKNOWN,
                confidence_score=0.0,
                possible_types=[],
                metadata={**metadata, "error": str(e)}
            )

