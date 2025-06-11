# main.py

import os
import json
import argparse
from src.data_processing import PDFProcessor
from src.feature_extraction import SemesterFeatureExtractor
from src.model import SemesterClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_pdf(pdf_path, model_path=None):
    """
    Process a single PDF and extract semester information.
    
    Args:
        pdf_path (str): Path to the PDF file
        model_path (str, optional): Path to the trained model. If None, use rule-based approach.
    """
    # Create processor
    processor = PDFProcessor()
    
    # Extract text from PDF
    extracted_texts = processor.extract_text_from_pdf(pdf_path)
    
    if not extracted_texts:
        logger.error(f"Failed to extract text from {pdf_path}")
        return []
    
    # Combine all pages' text
    combined_text = " ".join(extracted_texts)
    
    # Extract sentences
    sentences = processor.extract_sentences(combined_text)
    logger.info(f"Extracted {len(sentences)} sentences from {pdf_path}")
    
    # If model path provided, use trained model
    if model_path and os.path.exists(model_path):
        # Load the model
        classifier = SemesterClassifier.load(model_path)
        
        # Make predictions
        predictions = classifier.predict(sentences)
        probabilities = classifier.predict_proba(sentences)
        
        # Filter semester sentences
        semester_sentences = []
        for i, (sentence, pred, prob) in enumerate(zip(sentences, predictions, probabilities)):
            if pred == 1:
                semester_sentences.append({
                    'sentence': sentence,
                    'confidence': float(prob),
                    'original_index': i
                })
    else:
        # Use rule-based approach
        extractor = SemesterFeatureExtractor()
        features_list = extractor.extract_features_from_sentences(sentences)
        filtered_sentences = extractor.filter_semester_sentences(sentences, features_list)
        
        semester_sentences = []
        for item in filtered_sentences:
            semester_sentences.append({
                'sentence': item['sentence'],
                'confidence': 1.0 if item['features']['has_semester_pattern'] == 1 else 0.8,
                'original_index': item['original_index'],
                'semester_pattern': item['features'].get('semester_pattern_match')
            })
    
    logger.info(f"Identified {len(semester_sentences)} sentences with semester information")
    
    return semester_sentences

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract semester information from PDF files')
    parser.add_argument('pdf_path', help='Path to a PDF file or directory containing PDF files')
    parser.add_argument('--model', default='models/semester_classifier.pkl', help='Path to trained model')
    parser.add_argument('--output', default='results/semester_info.json', help='Output file path')
    args = parser.parse_args()
    
    # Process single PDF or directory
    if os.path.isfile(args.pdf_path):
        # Process single PDF
        logger.info(f"Processing PDF file: {args.pdf_path}")
        semester_sentences = process_pdf(args.pdf_path, args.model)
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({os.path.basename(args.pdf_path): semester_sentences}, f, indent=2)
        
        print(f"Processed 1 PDF file")
        print(f"Found {len(semester_sentences)} sentences with semester information")
        print(f"Results saved to {args.output}")
        
        # Print some examples
        if semester_sentences:
            print("\nExamples of semester information found:")
            for i, item in enumerate(semester_sentences[:5]):  # Show first 5
                print(f"  {i+1}. \"{item['sentence']}\" (confidence: {item['confidence']:.2f})")
            
            if len(semester_sentences) > 5:
                print(f"  ... and {len(semester_sentences) - 5} more")
        
    elif os.path.isdir(args.pdf_path):
        # Process all PDFs in directory
        pdf_files = [f for f in os.listdir(args.pdf_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.error(f"No PDF files found in {args.pdf_path}")
            return
        
        logger.info(f"Processing {len(pdf_files)} PDF files from directory: {args.pdf_path}")
        
        results = {}
        total_sentences = 0
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(args.pdf_path, pdf_file)
            semester_sentences = process_pdf(pdf_path, args.model)
            
            results[pdf_file] = semester_sentences
            total_sentences += len(semester_sentences)
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Processed {len(pdf_files)} PDF files")
        print(f"Found {total_sentences} sentences with semester information")
        print(f"Results saved to {args.output}")
        
        # Print some examples from each PDF
        if results:
            print("\nExamples of semester information found:")
            for pdf_file, sentences in list(results.items())[:3]:  # Show first 3 PDFs
                if sentences:
                    print(f"\nFrom {pdf_file}:")
                    for i, item in enumerate(sentences[:2]):  # Show first 2 sentences
                        print(f"  {i+1}. \"{item['sentence']}\" (confidence: {item['confidence']:.2f})")
                    
                    if len(sentences) > 2:
                        print(f"  ... and {len(sentences) - 2} more")
    else:
        logger.error(f"Path not found: {args.pdf_path}")

if __name__ == "__main__":
    main()