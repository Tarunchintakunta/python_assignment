# quick_test.py

import sys
import re
from src.data_processing import PDFProcessor
from src.feature_extraction import SemesterFeatureExtractor
from src.model import SemesterClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def extract_semester_format(sentence, pattern_match=None):
    """Extract the semester format from a sentence."""
    if pattern_match:
        return pattern_match
    
    # If no pattern match is provided, try to extract a semester format
    patterns = [
        r'(?:I|II|III|IV|V|VI|VII|VIII)\s+Year\s+(?:I|II|III|IV|V|VI|VII|VIII)\s+Semester',
        r'(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)\s+[Ss]emester',
        r'(?:1st|2nd|3rd|4th|5th|6th|7th|8th)\s+[Ss]emester',
        r'[Ss]emester\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|1|2|3|4|5|6|7|8)',
        r'S[1-8]',
        r'SEM\s*[:-]?\s*(?:I|II|III|IV|V|VI|VII|VIII)',
        r'SEMESTER\s*[,]?\s*[1-8]',
        r'B\.Tech\s+(?:I|II|III|IV|V|VI|VII|VIII)\s+Year\s+(?:I|II|III|IV|V|VI|VII|VIII)\s+Semester'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0)
    
    # If no specific pattern matched but contains "semester"
    if "semester" in sentence.lower():
        words = sentence.split()
        for i, word in enumerate(words):
            if 'semester' in word.lower():
                start = max(0, i-2)
                end = min(len(words), i+3)
                return ' '.join(words[start:end])
    
    return None

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
    
    # Create true labels array (1 for semester sentences, 0 for others)
    true_labels = [1 if i in rule_indices else 0 for i in range(len(sentences))]
    
    # Get model predictions
    classifier = SemesterClassifier.load("models/semester_classifier.pkl")
    predictions = classifier.predict(sentences)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    # Handle case where there are no positive examples
    if sum(true_labels) == 0 and sum(predictions) == 0:
        # If both ground truth and predictions have no positive examples,
        # then precision, recall, and F1 are technically undefined but considered 1.0
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    elif sum(true_labels) == 0 or sum(predictions) == 0:
        # If either has no positives, then at least one metric is 0
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
    else:
        # Normal case
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
    
    print(f"\nPDF: {pdf_path}")
    print(f"  Total sentences: {len(sentences)}")
    print(f"  Ground truth semester sentences: {sum(true_labels)}")
    print(f"  Predicted semester sentences: {sum(predictions)}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Extract all unique semester formats found
    semester_formats = set()
    for item in rule_results:
        pattern_match = item['features'].get('semester_pattern_match')
        format_text = extract_semester_format(item['sentence'], pattern_match)
        if format_text:
            semester_formats.add(format_text)
    
    # Print all semester formats detected in this PDF
    print("\nSemester Formats Detected in this PDF:")
    if semester_formats:
        for i, format_text in enumerate(sorted(semester_formats), 1):
            print(f"  {i}. \"{format_text}\"")
    else:
        print("  No semester formats detected")
    
    # Print correct and incorrect predictions
    true_positives = []
    false_negatives = []
    false_positives = []
    
    for i in range(len(sentences)):
        if i in rule_indices:  # Ground truth positive
            if predictions[i] == 1:  # True positive
                true_positives.append(i)
            else:  # False negative
                false_negatives.append(i)
        else:  # Ground truth negative
            if predictions[i] == 1:  # False positive
                false_positives.append(i)
    
    # # Print sample sentences
    # if true_positives:
    #     print("\nCorrectly Identified Semester Sentences (sample):")
    #     for i, idx in enumerate(true_positives[:3]):  # Show first 3
    #         print(f"  {i+1}. \"{sentences[idx]}\"")
    
    # if false_negatives:
    #     print("\nMissed Semester Sentences (false negatives):")
    #     for i, idx in enumerate(false_negatives[:3]):  # Show first 3
    #         print(f"  {i+1}. \"{sentences[idx]}\"")
    
    # if false_positives:
    #     print("\nIncorrectly Identified Sentences (false positives):")
    #     for i, idx in enumerate(false_positives[:3]):  # Show first 3
    #         print(f"  {i+1}. \"{sentences[idx]}\"")

if __name__ == "__main__":
    main()