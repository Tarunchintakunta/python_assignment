import json
import os
import re

def extract_semester_format(sentence):
    """Extract just the semester format from a sentence."""
    patterns = [
        r'(?:I|II|III|IV|V|VI|VII|VIII)\s+Year\s+(?:I|II|III|IV|V|VI|VII|VIII)\s+Semester',
        r'(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)\s+Semester',
        r'(?:1st|2nd|3rd|4th|5th|6th|7th|8th)\s+[Ss]emester',
        r'[Ss]emester\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|1|2|3|4|5|6|7|8)',
        r'S[1-8]',
        r'Sem-(?:I|II|III|IV|V|VI|VII|VIII)',
        r'SEMESTER\s*[1-8]',
        r'B\.Tech\s+(?:I|II|III|IV|V|VI|VII|VIII)\s+Year\s+(?:I|II|III|IV|V|VI|VII|VIII)\s+Semester'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0)
    
    # If no specific pattern matched, return a short excerpt
    words = sentence.split()
    for i, word in enumerate(words):
        if 'semester' in word.lower() or 'sem' in word.lower():
            start = max(0, i-2)
            end = min(len(words), i+3)
            return ' '.join(words[start:end])
    
    return None

def main():
    # Load test data (ground truth from rule-based approach)
    results_dir = "results"
    test_semester_path = os.path.join(results_dir, "test_semester_sentences.json")
    
    if not os.path.exists(test_semester_path):
        print("ERROR: Ground truth data not found. Run analyze_semester_info.py first.")
        return
    
    with open(test_semester_path, "r") as f:
        test_semester_data = json.load(f)
    
    print("Semester Formats Detected:\n")
    
    # Track unique formats to avoid repetition
    unique_formats = set()
    
    for pdf_name, sentences in test_semester_data.items():
        for item in sentences:
            semester_format = extract_semester_format(item['sentence'])
            if semester_format and semester_format not in unique_formats:
                unique_formats.add(semester_format)
    
    # Print in a clean, numbered list
    for i, format_text in enumerate(sorted(unique_formats), 1):
        print(f"{i}. \"{format_text}\"")

if __name__ == "__main__":
    main()