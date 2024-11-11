from typing import List, Dict, Tuple
from pydantic import BaseModel
from enum import Enum


class DocumentType(str, Enum):
    DRIVERS_LICENSE = "drivers_license"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"
    PAYSLIP = "payslip"
    TAX_RETURN = "tax_return"
    UTILITY_BILL = "utility_bill"
    UNKNOWN = "unknown"

class DocumentResponse(BaseModel):
    document_type: DocumentType
    confidence_score: float
    possible_types: List[Tuple[DocumentType, float]]
    metadata: Dict
    # processing_time: float

# class DocumentBatch(BaseModel):
#     documents: List[Dict]
#     batch_size: int
#     processing_time: float
