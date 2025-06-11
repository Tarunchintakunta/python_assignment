# quick_test.py

import sys
from src.data_processing import PDFProcessor
from src.feature_extraction import SemesterFeatureExtractor
from src.model import SemesterClassifier

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py path/to/pdf.pdf")
        return
    
    pdf_path = sys.argv[1]
    
    # Process the PDF
    processor = PDFProcessor()
    texts = processor.extract_text_from_pdf(pdf_path)
    combined_text = " ".join(texts)
    sentences = processor.extract_sentences(combined_text)
    
    print(f"Extracted {len(sentences)} sentences from {pdf_path}")
    
    # Get ground truth using rule-based approach
    extractor = SemesterFeatureExtractor()
    features = extractor.extract_features_from_sentences(sentences)
    rule_results = extractor.filter_semester_sentences(sentences, features)
    
    rule_indices = [item['original_index'] for item in rule_results]
    
    # Get model predictions
    classifier = SemesterClassifier.load("models/semester_classifier.pkl")
    predictions = classifier.predict(sentences)
    
    # Compare results
    match_count = 0
    for i, pred in enumerate(predictions):
        if (pred == 1 and i in rule_indices) or (pred == 0 and i not in rule_indices):
            match_count += 1
    
    accuracy = match_count / len(sentences)
    
    print(f"Rule-based approach found {len(rule_results)} semester sentences")
    print(f"Model predicted {sum(predictions)} semester sentences")
    print(f"Accuracy (compared to rule-based): {accuracy:.4f}")

if __name__ == "__main__":
    main()