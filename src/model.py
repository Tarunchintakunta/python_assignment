# src/model.py

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import logging
from src.feature_extraction import SemesterFeatureTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemesterClassifier:
    """
    Class for classifying sentences as containing semester information or not.
    """
    
    def __init__(self):
        """
        Initialize the semester classifier.
        """
        # Create a pipeline with both text features and custom semester features
        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text', Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), 
                                            min_df=2, use_idf=True, sublinear_tf=True))
                ])),
                ('semester', SemesterFeatureTransformer())
            ])),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, 
                                                class_weight='balanced', max_depth=10))
        ])
        
        self.is_trained = False
    
    def prepare_data(self, semester_sentences, regular_sentences):
        """
        Prepare labeled data for training.
        
        Args:
            semester_sentences (list): List of sentences containing semester information
            regular_sentences (list): List of sentences not containing semester information
            
        Returns:
            tuple: (X, y) where X is a list of sentences and y is a list of labels
        """
        # Combine sentences
        X = semester_sentences + regular_sentences
        
        # Create labels (1 for semester sentences, 0 for others)
        y = [1] * len(semester_sentences) + [0] * len(regular_sentences)
        
        return X, y
    
    def train(self, X, y):
        """
        Train the classifier.
        
        Args:
            X (list): List of sentences
            y (list): List of labels (1 for semester sentences, 0 for others)
        """
        logger.info(f"Training classifier on {len(X)} sentences...")
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.pipeline.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        
        # Print evaluation metrics
        logger.info(f"Validation accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_val, y_pred))
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, sentences):
        """
        Predict semester information in sentences.
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            list: List of predictions (1 for semester sentences, 0 for others)
        """
        if not self.is_trained:
            logger.warning("Model is not trained yet!")
            return []
        
        return self.pipeline.predict(sentences)
    
    def predict_proba(self, sentences):
        """
        Predict probability of sentences containing semester information.
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            list: List of probability estimates
        """
        if not self.is_trained:
            logger.warning("Model is not trained yet!")
            return []
        
        return self.pipeline.predict_proba(sentences)[:, 1]  # Probability of positive class
    
    def save(self, model_path):
        """
        Save the trained model to a file.
        
        Args:
            model_path (str): Path to save the model
        """
        if not self.is_trained:
            logger.warning("Attempting to save untrained model!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path):
        """
        Load a trained model from a file.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            SemesterClassifier: Loaded classifier
        """
        instance = cls()
        
        with open(model_path, 'rb') as f:
            instance.pipeline = pickle.load(f)
        
        instance.is_trained = True
        logger.info(f"Model loaded from {model_path}")
        
        return instance