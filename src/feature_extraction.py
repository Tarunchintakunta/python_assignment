import re
import numpy as np
import pandas as pd
import logging

import spacy
from sklearn.base import BaseEstimator, TransformerMixin


#configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# load spaCy model

try:
    nlp = spacy.load('en_core_web_sm')
    logger.info("Loaded spaCy model successfully")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    nlp = None

class SemesterFeatureExtractor:
    """
    Class for extracting features related to semester information from text.
    """
    
    def __init__(self):
        """
        Initialize the semester feature extractor.
        """
        # Regular expressions for identifying semester-related patterns
        self.semester_patterns = [
            r'\b(?:spring|summer|fall|winter)\s+(?:semester|term)?\s+(?:20\d{2})\b',
            r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth)\s+semester\b',
            r'\b(?:1st|2nd|3rd|4th|5th|6th|7th|8th)\s+semester\b',
            r'\bsemester\s+(?:one|two|three|four|five|six|seven|eight|1|2|3|4|5|6|7|8)\b',
            r'\bs[1-8]\b',  # Abbreviated form like S1, S2, etc.
            r'\bsem\s*[1-8]\b',  # Abbreviated form like Sem1, Sem 2, etc.
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:20\d{2})\s+semester\b',
            r'SEMESTER\s*[,]?\s*[1-8]',  # Common format in educational documents
            r'SEM\s*[,]?\s*[1-8]',  # Abbreviated format
            r'SEMESTER\s*[:-]?\s*(?:I|II|III|IV|V|VI|VII|VIII)',  # Roman numeral format
            r'SEM\s*[:-]?\s*(?:I|II|III|IV|V|VI|VII|VIII)'  # Abbreviated with Roman numerals
        ]
        
        # Compile the patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.semester_patterns]
    
    def extract_features_from_sentence(self, sentence):
        """
        Extract semester-related features from a sentence.
        
        Args:
            sentence (str): Input sentence
            
        Returns:
            dict: Dictionary of features
        """
        features = {
            'has_semester_keyword': 0,
            'has_semester_pattern': 0,
            'semester_position': -1,
            'contains_number': 0,
            'contains_month': 0,
            'contains_year': 0,
            'length': len(sentence),
            'semester_pattern_match': None,
            'is_all_caps': 1 if sentence.isupper() else 0  # Educational documents often have headers in all caps
        }
        
        # Check for semester keyword
        if re.search(r'\b(?:semester|sem|term)\b', sentence, re.IGNORECASE):
            features['has_semester_keyword'] = 1
        
        # Check for numbers in the sentence
        if re.search(r'\b\d+\b', sentence):
            features['contains_number'] = 1
        
        # Check for month names
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 
                 'august', 'september', 'october', 'november', 'december',
                 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        if any(re.search(r'\b' + month + r'\b', sentence, re.IGNORECASE) for month in months):
            features['contains_month'] = 1
        
        # Check for years (assuming 2000-2099)
        if re.search(r'\b20\d{2}\b', sentence):
            features['contains_year'] = 1
        
        # Check for semester patterns
        for i, pattern in enumerate(self.compiled_patterns):
            match = pattern.search(sentence)
            if match:
                features['has_semester_pattern'] = 1
                features['semester_position'] = match.start() / len(sentence)  # Normalized position
                features['semester_pattern_match'] = match.group(0)
                break
        
        # Use spaCy for additional NLP features if available
        if nlp:
            doc = nlp(sentence)
            
            # Count entities of interest
            entity_counts = {}
            for ent in doc.ents:
                entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
            
            # Add entity counts as features
            for label, count in entity_counts.items():
                features[f'entity_{label}'] = count
            
            # Check for educational terms
            edu_terms = ['grade', 'course', 'exam', 'credit', 'gpa', 'curriculum', 'syllabus', 'study']
            features['contains_edu_terms'] = 1 if any(term in sentence.lower() for term in edu_terms) else 0
        
        return features
    
    def extract_features_from_sentences(self, sentences):
        """
        Extract features from a list of sentences.
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            list: List of feature dictionaries
        """
        features_list = []
        
        for sentence in sentences:
            features = self.extract_features_from_sentence(sentence)
            features_list.append(features)
        
        return features_list
    
    def filter_semester_sentences(self, sentences, features_list=None):
        """
        Filter sentences to only include those that are likely to contain semester information.
        
        Args:
            sentences (list): List of sentences
            features_list (list, optional): Pre-computed features. If None, features will be computed.
            
        Returns:
            list: List of filtered sentences that likely contain semester information
        """
        if features_list is None:
            features_list = self.extract_features_from_sentences(sentences)
        
        semester_sentences = []
        
        for i, (sentence, features) in enumerate(zip(sentences, features_list)):
            # Simple rule-based filtering
            if features['has_semester_keyword'] == 1 or features['has_semester_pattern'] == 1:
                semester_sentences.append({
                    'sentence': sentence,
                    'features': features,
                    'original_index': i
                })
        
        return semester_sentences


class SemesterFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for extracting semester-related features for use in scikit-learn pipelines.
    """
    
    def __init__(self):
        self.extractor = SemesterFeatureExtractor()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Transform a list of sentences into a feature matrix.
        
        Args:
            X (list): List of sentences
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        features_list = self.extractor.extract_features_from_sentences(X)
        
        # Convert to numpy array format
        feature_names = ['has_semester_keyword', 'has_semester_pattern', 'semester_position', 
                        'contains_number', 'contains_month', 'contains_year', 'length', 
                        'is_all_caps']
        
        feature_matrix = np.zeros((len(X), len(feature_names)))
        
        for i, features in enumerate(features_list):
            for j, name in enumerate(feature_names):
                feature_matrix[i, j] = features.get(name, 0)
        
        return feature_matrix


# Example usage
if __name__ == "__main__":
    extractor = SemesterFeatureExtractor()
    
    # Example sentences
    test_sentences = [
        "This course is offered in Spring 2023.",
        "Students must complete this by the end of the 3rd semester.",
        "The deadline for Fall Semester 2022 registration is August 15.",
        "This document has nothing to do with semesters.",
        "In S1 2023, new courses will be available.",
        "Sem 4 will focus on advanced topics.",
        "SEMESTER , REGULAR EXAMINATIONS Branch COMPUTER SCIENCE",
        "CONSOLIDATED GRADE RECORD (CBCS) SEM - III"
    ]
    
    # Extract features
    features = extractor.extract_features_from_sentences(test_sentences)
    
    # Filter semester sentences
    semester_sentences = extractor.filter_semester_sentences(test_sentences, features)
    
    # Print results
    print(f"Detected {len(semester_sentences)} sentences with semester information:")
    for item in semester_sentences:
        print(f"- {item['sentence']}")
        print(f"  Match: {item['features'].get('semester_pattern_match', 'None')}")
        print()