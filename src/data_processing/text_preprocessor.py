import re
import string
import nltk
import spacy
from nltk.corpus import stopwords

class TextPreprocessor:
    """
    Enhanced text preprocessing pipeline for NewsBot 2.0.
    Handles cleaning, stopword removal, and lemmatization. [cite: 1538]
    """
    
    def __init__(self, spacy_model="en_core_web_sm"):
        """Initialize NLP resources and load models. [cite: 331, 332]"""
        try:
            self.nlp = spacy.load(spacy_model) [cite: 341]
        except OSError:
            # Automatic fallback if model isn't installed
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(spacy_model)
            
        # Initialize stopwords from NLTK
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def clean_raw_text(self, text):
        """
        Performs basic string cleaning: lowercasing, URL removal, and punctuation stripping.
        Extracted from Midterm logic.
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove URLs [cite: 356]
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 3. Remove punctuation and special characters [cite: 358]
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        
        # 4. Remove extra whitespace [cite: 356]
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process(self, text, use_lemmatization=True):
        """
        Main pipeline to transform raw text into a cleaned, tokenized format.
        Returns a single string of processed tokens. 
        """
        # Clean the string first
        cleaned = self.clean_raw_text(text)
        
        if not cleaned:
            return ""

        # Process with spaCy for lemmatization and stopword removal [cite: 341, 353]
        doc = self.nlp(cleaned)
        
        tokens = []
        for token in doc:
            # Check if token is a stopword or too short
            if token.text not in self.stop_words and len(token.text) > 2:
                if use_lemmatization:
                    tokens.append(token.lemma_)
                else:
                    tokens.append(token.text)
                    
        return " ".join(tokens)

    def batch_process(self, texts):
        """Processes a list of strings efficiently. [cite: 1510]"""
        return [self.process(t) for t in texts]

if __name__ == "__main__":
    # Quick test for Phase 1 validation [cite: 1510]
    preprocessor = TextPreprocessor()
    sample = "Apple Inc. is looking at buying a U.K. startup for $1 billion! https://apple.com"
    print(f"Original: {sample}")
    print(f"Processed: {preprocessor.process(sample)}")