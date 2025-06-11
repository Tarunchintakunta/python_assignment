
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from src.model import SemesterClassifier

def main():
    # Load test data (ground truth from rule-based approach)
    with open("results/test_semester_sentences.json", "r") as f:
        test_semester_data = json.load(f)
    
    # Load all extracted test sentences
    with open("results/extracted_test_sentences.json", "r") as f:
        test_all_data = json.load(f)
    
    # Load the trained model
    model_path = "models/semester_classifier.pkl"
    classifier = SemesterClassifier.load(model_path)
    
    # Process each PDF
    print("Evaluating model on test PDFs:")
    
    overall_true = []
    overall_pred = []
    
    for pdf_name, sentences in test_all_data.items():
        # Prepare data for this PDF
        true_labels = []
        
        # Get the indices of semester sentences for this PDF (ground truth)
        semester_indices = []
        if pdf_name in test_semester_data:
            semester_indices = [item['original_index'] for item in test_semester_data[pdf_name]]
        
        # Create true labels (1 for semester sentences, 0 for others)
        for i in range(len(sentences)):
            true_labels.append(1 if i in semester_indices else 0)
        
        # Make predictions
        predictions = classifier.predict(sentences)
        
        # Add to overall metrics
        overall_true.extend(true_labels)
        overall_pred.extend(predictions)
        
        # Calculate metrics for this PDF
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        # Count true positives and predicted positives
        true_positives = sum(true_labels)
        predicted_positives = sum(predictions)
        
        print(f"\nPDF: {pdf_name}")
        print(f"  Total sentences: {len(sentences)}")
        print(f"  Ground truth semester sentences: {true_positives}")
        print(f"  Predicted semester sentences: {predicted_positives}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(overall_true, overall_pred)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        overall_true, overall_pred, average='binary', zero_division=0
    )
    
    print("\nOverall Test Set Metrics:")
    print(f"  Total PDFs: {len(test_all_data)}")
    print(f"  Total sentences: {len(overall_true)}")
    print(f"  Ground truth semester sentences: {sum(overall_true)}")
    print(f"  Predicted semester sentences: {sum(overall_pred)}")
    print(f"  Accuracy: {overall_accuracy:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1 Score: {overall_f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(overall_true, overall_pred))

if __name__ == "__main__":
    main()