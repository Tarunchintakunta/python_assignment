import os
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        """
        Initialize the PDF processor.
        For macOS, Tesseract and Poppler paths are usually not needed if installed via Homebrew.
        """
        # Test if Tesseract is installed and accessible
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is properly installed and accessible.")
        except Exception as e:
            logger.error(f"Tesseract OCR is not properly configured: {str(e)}")
            logger.error("Please ensure Tesseract is installed using 'brew install tesseract'")
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from all pages of a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of extracted text strings, one per page
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Convert PDF to images
            # On macOS with Homebrew, poppler_path is not needed
            images = convert_from_path(
                pdf_path,
                dpi=300  # Higher DPI for better OCR results
            )
            
            logger.info(f"Converted PDF to {len(images)} images")
            
            # Extract text from each image
            extracted_texts = []
            for i, image in enumerate(images):
                logger.info(f"Performing OCR on page {i+1}/{len(images)}")
                
                # Convert image to text using Tesseract
                text = pytesseract.image_to_string(image, lang='eng')
                
                # Clean the extracted text
                text = self._clean_text(text)
                
                extracted_texts.append(text)
                
            return extracted_texts
        
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
    
    def _clean_text(self, text):
        """
        Clean and preprocess the extracted text.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might be OCR artifacts
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text

    def extract_sentences(self, text):
        """
        Split text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        # Simple sentence splitting by common sentence terminators
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    def process_pdf_directory(self, directory_path):
        """
        Process all PDFs in a directory and extract text.
        
        Args:
            directory_path (str): Path to directory containing PDFs
            
        Returns:
            dict: Dictionary with PDF filenames as keys and lists of extracted sentences as values
        """
        results = {}
        
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return results
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        # Process each PDF file
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            
            # Extract text from the PDF
            extracted_texts = self.extract_text_from_pdf(pdf_path)
            
            # Combine all pages' text
            combined_text = " ".join(extracted_texts)
            
            # Extract sentences
            sentences = self.extract_sentences(combined_text)
            
            # Store results
            results[pdf_file] = sentences
        
        return results


# Example usage
if __name__ == "__main__":
    # Create processor instance (no paths needed for macOS)
    processor = PDFProcessor()
    
    # Process a single PDF
    sample_pdf = "../data/train/sample.pdf"
    if os.path.exists(sample_pdf):
        texts = processor.extract_text_from_pdf(sample_pdf)
        print(f"Extracted {len(texts)} pages of text")
        
        # Print first 200 characters of first page
        if texts:
            print(f"Sample text: {texts[0][:200]}...")
    
    # Process all PDFs in training directory
    train_dir = "../data/train"
    if os.path.exists(train_dir):
        results = processor.process_pdf_directory(train_dir)
        print(f"Processed {len(results)} PDF files")