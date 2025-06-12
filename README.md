# Python Assignment
# Semester Information Recognition

An AI model to extract and categorize semester information from PDF files containing images.

## Overview
This project uses OCR (Tesseract) and Named Entity Recognition techniques to:
1. Extract text from images contained in PDF files
2. Identify and categorize semester-related sentences
3. Achieve at least 50% accuracy in classification

## Dependencies
- Python 3.8+
- Tesseract OCR
- Poppler
- See requirements.txt for Python package dependencies

## Installation

1. Install Tesseract OCR:
   ```
   # macOS (using Homebrew)
   brew install tesseract
   
   # Linux
   sudo apt install tesseract-ocr
   ```

2. Install Poppler:
   ```
   # macOS
   brew install poppler
   
   # Linux
   sudo apt install poppler-utils
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

4. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Download required NLTK and spaCy resources:
   ```
   python download_nltk.py
   python -m spacy download en_core_web_sm
   ```

## Project Structure

```
semester_recognition_project/
├── data/
│   ├── train/        # Training PDF files
│   └── test/         # Test PDF files
├── models/           # Saved models
├── results/          # Output files
├── src/              # Source code
│   ├── __init__.py
│   ├── data_processing.py    # PDF processing and OCR
│   ├── feature_extraction.py # Feature extraction
│   ├── model.py              # Machine learning model
├── main.py                   # Main application
├── text_extraction.py        # Script to extract text from PDFs
├── analyze_semester_info.py  # Script to analyze extracted text
├── train_model.py            # Script to train and evaluate the model
├── quick_test.py             # Script to test on a single PDF
├── evaluate_test_pdfs.py     # Script to evaluate on all test PDFs
├── README.md                 # Documentation
└── requirements.txt          # Dependencies list
```

## Usage

### Processing PDFs

1. Extract text from PDFs:
   ```
   python text_extraction.py
   ```

2. Analyze text for semester information:
   ```
   python analyze_semester_info.py
   ```

3. Train the model:
   ```
   python train_model.py
   ```

4. Process a single PDF:
   ```
   python main.py path/to/your/file.pdf
   ```

5. Process all PDFs in a directory:
   ```
   python main.py path/to/your/directory
   ```

### Testing and Evaluation

1. Quick test on a single PDF:
   ```
   python quick_test.py path/to/your/file.pdf
   ```

2. Evaluate performance on all test PDFs:
   ```
   python evaluate_test_pdfs.py
   ```

## Performance

The system achieves excellent performance on test data:
- **Accuracy**: 93.72%
- **Precision**: 100.00%
- **Recall**: 73.33% 
- **F1 Score**: 84.62%

Individual file performance:
- Application_884203492.pdf: 96.83% accuracy
- Application_109341359.pdf: 100.00% accuracy 
- Application_981109807.pdf: 83.61% accuracy

## Semester Format Recognition

The system can identify a wide range of semester formats, including:
- "I Year I Semester"
- "First Semester"
- "Semester 1"
- "B.Tech I Year I Semester (R16) Reg."
- "SEMESTER 11"
- "S8" (abbreviated form)
- And many more variations

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 
  - Text-based (TF-IDF)
  - Semester pattern matching
  - Contextual features (numbers, dates, etc.)
- **Parameters**:
  - n_estimators=100
  - max_depth=10
  - class_weight='balanced'

The model significantly exceeds the required 50% accuracy threshold, demonstrating effective semester information extraction from PDF documents.