# train_model.py

import os
import json
import random
import numpy as np
from src.model import SemesterClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load the semester sentences
    results_dir = "results"
    train_semester_path = os.path.join(results_dir, "train_semester_sentences.json")
    
    if not os.path.exists(train_semester_path):
        print("ERROR: Semester sentences file not found. Run analyze_semester_info.py first.")
        return
    
    with open(train_semester_path, "r") as f:
        train_semester_data = json.load(f)
    
    # Load all extracted sentences
    train_all_path = os.path.join(results_dir, "extracted_train_sentences.json")
    
    with open(train_all_path, "r") as f:
        train_all_data = json.load(f)
    
    # Prepare training data
    semester_sentences = []
    for pdf_name, sentences in train_semester_data.items():
        for item in sentences:
            semester_sentences.append(item['sentence'])
    
    # Get non-semester sentences
    non_semester_sentences = []
    for pdf_name, sentences in train_all_data.items():
        # Get the semester sentence indices for this PDF
        semester_indices = []
        if pdf_name in train_semester_data:
            semester_indices = [item['original_index'] for item in train_semester_data[pdf_name]]
        
        # Add non-semester sentences
        for i, sentence in enumerate(sentences):
            if i not in semester_indices:
                non_semester_sentences.append(sentence)
    
    print(f"Found {len(semester_sentences)} semester sentences")
    print(f"Found {len(non_semester_sentences)} non-semester sentences")
    
    # Balance the dataset (optional)
    if len(non_semester_sentences) > 2 * len(semester_sentences):
        non_semester_sentences = random.sample(non_semester_sentences, 2 * len(semester_sentences))
        print(f"Sampled {len(non_semester_sentences)} non-semester sentences to balance the dataset")
    
    # Create and train the model
    classifier = SemesterClassifier()
    X, y = classifier.prepare_data(semester_sentences, non_semester_sentences)
    
    print(f"Training model on {len(X)} sentences ({sum(y)} positive, {len(y) - sum(y)} negative)...")
    metrics = classifier.train(X, y)
    
    # Save the model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "semester_classifier.pkl")
    classifier.save(model_path)
    
    # Test on the test data
    test_path = os.path.join(results_dir, "extracted_test_sentences.json")
    
    with open(test_path, "r") as f:
        test_data = json.load(f)
    
    # Flatten test sentences
    test_sentences = []
    pdf_map = []  # Keep track of which PDF and index each sentence comes from
    
    for pdf_name, sentences in test_data.items():
        for i, sentence in enumerate(sentences):
            test_sentences.append(sentence)
            pdf_map.append((pdf_name, i))
    
    # Make predictions
    print(f"\nMaking predictions on {len(test_sentences)} test sentences...")
    predictions = classifier.predict(test_sentences)
    probabilities = classifier.predict_proba(test_sentences)
    
    # Organize predictions by PDF
    test_predictions = {}
    for (pdf_name, idx), pred, prob, sentence in zip(pdf_map, predictions, probabilities, test_sentences):
        if pdf_name not in test_predictions:
            test_predictions[pdf_name] = []
        
        if pred == 1:  # Only include predicted semester sentences
            test_predictions[pdf_name].append({
                'sentence': sentence,
                'probability': float(prob),
                'original_index': idx
            })
    
    # Save predictions
    with open(os.path.join(results_dir, "test_predictions.json"), "w") as f:
        json.dump(test_predictions, f, indent=2)
    
    # Print results
    total_predicted = sum(predictions)
    print(f"\nPredicted {total_predicted} semester sentences in test data")
    
    for pdf_name, predictions in test_predictions.items():
        print(f"\nPredictions for {pdf_name}:")
        for i, pred in enumerate(predictions[:5]):  # Show first 5
            print(f"  {i+1}. \"{pred['sentence']}\" (confidence: {pred['probability']:.2f})")
        
        if len(predictions) > 5:
            print(f"  ... and {len(predictions) - 5} more")
    
    print("\nTraining metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nModel saved to {model_path}")
    print(f"Predictions saved to {os.path.join(results_dir, 'test_predictions.json')}")

if __name__ == "__main__":
    main()