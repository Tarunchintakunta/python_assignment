# analyze_semester_info.py

import os
import json
from src.feature_extraction import SemesterFeatureExtractor

def main():
    # Create feature extractor
    extractor = SemesterFeatureExtractor()
    
    # Load extracted text
    results_dir = "results"
    train_path = os.path.join(results_dir, "extracted_train_sentences.json")
    test_path = os.path.join(results_dir, "extracted_test_sentences.json")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("ERROR: Extracted sentence files not found. Run text_extraction.py first.")
        return
    
    with open(train_path, "r") as f:
        train_data = json.load(f)
    
    with open(test_path, "r") as f:
        test_data = json.load(f)
    
    # Analyze training data
    print("Analyzing training data for semester information...")
    train_semester_sentences = {}
    total_train_sentences = 0
    total_train_semester_sentences = 0
    
    for pdf_name, sentences in train_data.items():
        # Extract features
        features_list = extractor.extract_features_from_sentences(sentences)
        
        # Filter semester sentences
        semester_sentences = extractor.filter_semester_sentences(sentences, features_list)
        
        train_semester_sentences[pdf_name] = semester_sentences
        total_train_sentences += len(sentences)
        total_train_semester_sentences += len(semester_sentences)
        
        print(f"\nPDF: {pdf_name}")
        print(f"  Found {len(semester_sentences)} semester sentences out of {len(sentences)} total sentences")
        
        # Print sample of semester sentences
        for i, item in enumerate(semester_sentences[:3]):  # First 3
            print(f"  {i+1}. {item['sentence']}")
            match = item['features'].get('semester_pattern_match')
            if match:
                print(f"     Match: '{match}'")
        
        if len(semester_sentences) > 3:
            print(f"  ... and {len(semester_sentences) - 3} more")
    
    # Analyze test data
    print("\nAnalyzing test data for semester information...")
    test_semester_sentences = {}
    total_test_sentences = 0
    total_test_semester_sentences = 0
    
    for pdf_name, sentences in test_data.items():
        # Extract features
        features_list = extractor.extract_features_from_sentences(sentences)
        
        # Filter semester sentences
        semester_sentences = extractor.filter_semester_sentences(sentences, features_list)
        
        test_semester_sentences[pdf_name] = semester_sentences
        total_test_sentences += len(sentences)
        total_test_semester_sentences += len(semester_sentences)
        
        print(f"\nPDF: {pdf_name}")
        print(f"  Found {len(semester_sentences)} semester sentences out of {len(sentences)} total sentences")
        
        # Print sample of semester sentences
        for i, item in enumerate(semester_sentences[:3]):  # First 3
            print(f"  {i+1}. {item['sentence']}")
            match = item['features'].get('semester_pattern_match')
            if match:
                print(f"     Match: '{match}'")
        
        if len(semester_sentences) > 3:
            print(f"  ... and {len(semester_sentences) - 3} more")
    
    # Save results
    train_output = {}
    for pdf_name, sentences in train_semester_sentences.items():
        train_output[pdf_name] = [
            {
                'sentence': item['sentence'],
                'original_index': item['original_index'],
                'semester_pattern_match': item['features'].get('semester_pattern_match')
            }
            for item in sentences
        ]
    
    test_output = {}
    for pdf_name, sentences in test_semester_sentences.items():
        test_output[pdf_name] = [
            {
                'sentence': item['sentence'],
                'original_index': item['original_index'],
                'semester_pattern_match': item['features'].get('semester_pattern_match')
            }
            for item in sentences
        ]
    
    with open(os.path.join(results_dir, "train_semester_sentences.json"), "w") as f:
        json.dump(train_output, f, indent=2)
    
    with open(os.path.join(results_dir, "test_semester_sentences.json"), "w") as f:
        json.dump(test_output, f, indent=2)
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"  Training data: {total_train_semester_sentences} semester sentences out of {total_train_sentences} total sentences ({total_train_semester_sentences/total_train_sentences*100:.2f}%)")
    print(f"  Test data: {total_test_semester_sentences} semester sentences out of {total_test_sentences} total sentences ({total_test_semester_sentences/total_test_sentences*100:.2f}%)")
    print("\nResults saved to:")
    print(f"  - results/train_semester_sentences.json")
    print(f"  - results/test_semester_sentences.json")

if __name__ == "__main__":
    main()