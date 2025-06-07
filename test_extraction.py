import os
import json
from src.data_processing import PDFProcessor

def main():
    # Create processor instance (no paths needed for macOS)
    processor = PDFProcessor()
    
    # Process training PDFs
    train_dir = "data/train"
    print(f"Processing training PDFs from {train_dir}...")
    train_results = processor.process_pdf_directory(train_dir)
    
    # Process test PDFs
    test_dir = "data/test"
    print(f"\nProcessing test PDFs from {test_dir}...")
    test_results = processor.process_pdf_directory(test_dir)
    
    # Save the extracted sentences
    os.makedirs("results", exist_ok=True)
    
    with open("results/extracted_train_sentences.json", "w") as f:
        json.dump(train_results, f, indent=2)
    
    with open("results/extracted_test_sentences.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Print statistics
    total_train_sentences = sum(len(sentences) for sentences in train_results.values())
    total_test_sentences = sum(len(sentences) for sentences in test_results.values())
    
    print("\nExtraction Statistics:")
    print(f"  Processed {len(train_results)} training PDFs")
    print(f"  Extracted {total_train_sentences} sentences from training PDFs")
    print(f"  Processed {len(test_results)} test PDFs")
    print(f"  Extracted {total_test_sentences} sentences from test PDFs")
    
    # Print a sample of sentences from each PDF
    print("\nSample sentences from training PDFs:")
    for pdf_name, sentences in list(train_results.items())[:2]:  # First 2 PDFs
        print(f"\n{pdf_name}:")
        for i, sentence in enumerate(sentences[:3]):  # First 3 sentences
            print(f"  {i+1}. {sentence}")
        
        if len(sentences) > 3:
            print(f"  ... ({len(sentences) - 3} more sentences)")
    
    print("\nResults saved to:")
    print(f"  results/extracted_train_sentences.json")
    print(f"  results/extracted_test_sentences.json")

if __name__ == "__main__":
    main()