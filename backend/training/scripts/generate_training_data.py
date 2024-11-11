import os
from pathlib import Path
from faker import Faker
from fpdf import FPDF
import random
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil

fake = Faker()

class DocumentGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.samples_per_type = 100  # Number of samples per document type
        
        # Create output directories
        for doc_type in ["drivers_license", "bank_statement", "invoice", 
                        "payslip", "tax_return", "utility_bill"]:
            (self.output_dir / doc_type).mkdir(parents=True, exist_ok=True)

    def generate_drivers_license(self) -> str:
        """Generate driver's license text."""
        return f"""
DRIVER LICENSE
ID: {fake.random_number(digits=8)}
NAME: {fake.name()}
ADDRESS: {fake.address()}
DOB: {fake.date_of_birth().strftime('%m/%d/%Y')}
CLASS: {random.choice(['A', 'B', 'C', 'D'])}
ENDORSEMENTS: {random.choice(['NONE', 'M1', 'CDL'])}
RESTRICTIONS: {random.choice(['NONE', 'CORRECTIVE LENSES'])}
ISSUED: {fake.date_this_year().strftime('%m/%d/%Y')}
EXPIRES: {(fake.date_this_year() + timedelta(days=1460)).strftime('%m/%d/%Y')}
"""

    def generate_bank_statement(self) -> str:
        """Generate bank statement text."""
        transactions = []
        balance = random.uniform(1000, 10000)
        for _ in range(10):
            amount = random.uniform(-500, 500)
            balance += amount
            transactions.append(
                f"{fake.date_this_month()}\t{fake.company()}\t"
                f"${amount:.2f}\t${balance:.2f}"
            )
        
        return f"""
MONTHLY BANK STATEMENT
ACCOUNT: {fake.random_number(digits=10)}
PERIOD: {fake.date_this_month().strftime('%B %Y')}

TRANSACTIONS:
DATE\t\tDESCRIPTION\tAMOUNT\tBALANCE
{''.join(transactions)}

ENDING BALANCE: ${balance:.2f}
"""

    def generate_invoice(self) -> str:
        """Generate invoice text."""
        items = []
        total = 0
        for _ in range(random.randint(3, 7)):
            qty = random.randint(1, 5)
            price = random.uniform(10, 100)
            amount = qty * price
            total += amount
            items.append(
                f"{fake.word()}\t{qty}\t${price:.2f}\t${amount:.2f}"
            )
        
        return f"""
INVOICE
Invoice #: INV-{fake.random_number(digits=6)}
Date: {fake.date_this_month().strftime('%m/%d/%Y')}
Due Date: {(fake.date_this_month() + timedelta(days=30)).strftime('%m/%d/%Y')}

Bill To:
{fake.company()}
{fake.address()}

ITEMS:
Description\tQty\tPrice\tAmount
{''.join(items)}

Subtotal: ${total:.2f}
Tax (10%): ${(total * 0.1):.2f}
Total: ${(total * 1.1):.2f}
"""

    def generate_payslip(self) -> str:
        """Generate payslip text."""
        base_salary = random.uniform(3000, 8000)
        tax = base_salary * 0.2
        insurance = 200
        return f"""
PAYSLIP
Employee: {fake.name()}
Employee ID: {fake.random_number(digits=6)}
Period: {fake.date_this_month().strftime('%B %Y')}

Earnings:
Base Salary: ${base_salary:.2f}
Overtime: ${random.uniform(0, 500):.2f}

Deductions:
Tax: ${tax:.2f}
Insurance: ${insurance:.2f}

Net Pay: ${(base_salary - tax - insurance):.2f}
"""

    def generate_tax_return(self) -> str:
        """Generate tax return text."""
        income = random.uniform(40000, 120000)
        return f"""
TAX RETURN FORM 1040
Tax Year: {datetime.now().year - 1}

Taxpayer Information:
Name: {fake.name()}
SSN: XXX-XX-{fake.random_number(digits=4)}
Filing Status: {random.choice(['Single', 'Married Filing Jointly'])}

Income:
Wages: ${income:.2f}
Interest: ${random.uniform(0, 1000):.2f}
Dividends: ${random.uniform(0, 2000):.2f}

Deductions:
Standard Deduction: ${12950:.2f}
Total Deductions: ${12950:.2f}

Tax Calculation:
Taxable Income: ${(income - 12950):.2f}
Tax: ${(income - 12950) * 0.22:.2f}
"""

    def generate_utility_bill(self) -> str:
        """Generate utility bill text."""
        usage = random.uniform(500, 1500)
        rate = 0.12
        return f"""
UTILITY BILL
Account #: {fake.random_number(digits=10)}
Service Period: {fake.date_this_month().strftime('%B %Y')}

Customer Information:
{fake.name()}
{fake.address()}

Usage Details:
Current Reading: {usage:.2f} kWh
Rate per kWh: ${rate:.2f}
Current Charges: ${(usage * rate):.2f}

Previous Balance: ${random.uniform(0, 100):.2f}
Total Amount Due: ${(usage * rate + random.uniform(0, 100)):.2f}
Due Date: {(fake.date_this_month() + timedelta(days=21)).strftime('%m/%d/%Y')}
"""

    def create_image_with_text(self, text: str, filename: str):
        """Create an image with the given text."""
        # Create image
        width = 1000
        height = 1400
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Add some basic graphics
        draw.rectangle([50, 50, width-50, height-50], outline='black')
        
        # Add text
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
            
        draw.text((100, 100), text, fill='black', font=font)
        
        # Save image
        image.save(filename)

    def generate_all(self):
        """Generate all types of documents."""
        generators = {
            'drivers_license': self.generate_drivers_license,
            'bank_statement': self.generate_bank_statement,
            'invoice': self.generate_invoice,
            'payslip': self.generate_payslip,
            'tax_return': self.generate_tax_return,
            'utility_bill': self.generate_utility_bill
        }
        
        for doc_type, generator in generators.items():
            output_dir = self.output_dir / doc_type
            print(f"Generating {doc_type} documents...")
            
            for i in range(self.samples_per_type):
                # Generate text content
                text = generator()
                
                # Create both PDF and image versions
                # Image version
                img_path = output_dir / f"{doc_type}_{i}.png"
                self.create_image_with_text(text, str(img_path))
                
                # PDF version
                pdf_path = output_dir / f"{doc_type}_{i}.pdf"
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", size=12)
                pdf.multi_cell(0, 10, text)
                pdf.output(str(pdf_path))

if __name__ == "__main__":
    output_dir = Path("join-the-siege/backend/training/synthetic_data")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    generator = DocumentGenerator(output_dir)
    generator.generate_all()
    print("Training data generation complete!")