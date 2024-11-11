import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from enum import Enum

class DocumentType(str, Enum):
    DRIVERS_LICENSE = "drivers_license"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"
    PAYSLIP = "payslip"
    TAX_RETURN = "tax_return"
    UTILITY_BILL = "utility_bill"
    UNKNOWN = "unknown"

class DocumentTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Use DistilBERT instead of RoBERTa (smaller and faster)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(DocumentType)
        ).to(self.device)
        
        # Training settings
        self.max_length = 128  # Reduced from 512
        self.batch_size = 16
        self.epochs = 2
        
    def prepare_data(self):
        """Prepare small dataset for quick training."""
        data = []
        
        # Sample documents for each type
        samples = {
            DocumentType.DRIVERS_LICENSE: [
                "DRIVER LICENSE ID Number Class C Expiration Date",
                "STATE DRIVER LICENSE Name Address DOB",
                "Driver's License Number Restrictions Class Type"
            ],
            DocumentType.BANK_STATEMENT: [
                "MONTHLY STATEMENT Account Balance Transactions",
                "Bank Statement Period Account Number Available Balance",
                "Banking Summary Deposits Withdrawals Account Details"
            ],
            DocumentType.INVOICE: [
                "INVOICE Bill To Amount Due Payment Terms",
                "Invoice Number Date Due Total Amount Customer ID",
                "Sales Invoice Subtotal Tax Total Amount Due"
            ],
            DocumentType.PAYSLIP: [
                "PAYSLIP Employee ID Gross Pay Net Pay",
                "Salary Slip Employee Name Pay Period Deductions",
                "Payroll Statement Earnings Tax Benefits"
            ],
            DocumentType.TAX_RETURN: [
                "TAX RETURN Form 1040 Taxable Income",
                "Federal Tax Return Filing Status Deductions",
                "Tax Year Income Tax Refund Amount"
            ],
            DocumentType.UTILITY_BILL: [
                "UTILITY BILL Account Number Due Date",
                "Electricity Bill Current Charges Usage Details",
                "Gas and Electric Service Bill Payment Due"
            ]
        }
        
        # Create dataset
        for doc_type, texts in samples.items():
            for text in texts:
                data.append({
                    'text': text,
                    'label': doc_type
                })
                
                # Add variations with some noise
                for _ in range(2):
                    words = text.split()
                    np.random.shuffle(words)
                    data.append({
                        'text': ' '.join(words),
                        'label': doc_type
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Split into train/val
        train_df, val_df = train_test_split(df, test_size=0.2)
        
        # Create torch datasets
        train_data = self.create_dataset(train_df)
        val_data = self.create_dataset(val_df)
        
        return train_data, val_data
    
    def create_dataset(self, df):
        """Create a simple dataset."""
        encodings = self.tokenizer(
            df['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        labels = [list(DocumentType).index(label) for label in df['label']]
        labels = torch.tensor(labels)
        
        return [{
            'input_ids': encodings['input_ids'][i],
            'attention_mask': encodings['attention_mask'][i],
            'labels': labels[i]
        } for i in range(len(df))]
    
    def train(self):
        """Quick training loop."""
        print("Preparing data...")
        train_data, val_data = self.prepare_data()
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        print("Starting training...")
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += len(batch['labels'])
            
            accuracy = correct / total
            print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        # Save the model
        output_dir = Path("/src/trained_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        
        return {"final_loss": avg_loss, "final_accuracy": accuracy}

def main():
    trainer = DocumentTrainer()
    metrics = trainer.train()
    print(f"Training complete! Metrics: {metrics}")

if __name__ == "__main__":
    main()