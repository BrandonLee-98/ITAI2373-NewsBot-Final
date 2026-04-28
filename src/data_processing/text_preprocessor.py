import re
import spacy
from nltk.corpus import stopwords

class TextPreprocessor:
    """Handles cleaning, stopword removal, and lemmatization."""
    
    def __init__(self, spacy_model="en_core_web_sm"):
        """Initialize NLP resources and load models."""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            # Fallback if model isn't linked correctly
            import en_core_web_sm
            self.nlp = en_core_web_sm.load()
            
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Standard cleaning pipeline."""
        # 1. Lowercase
        text = text.lower()
        # 2. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # 3. Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # 4. Remove extra whitespace
        text = " ".join(text.split())
        return text

    def preprocess(self, text):
        """Full preprocessing: clean -> tokenize -> lemmatize -> remove stopwords."""
        cleaned = self.clean_text(text)
        
        # Process with spaCy for lemmatization and stopword removal
        doc = self.nlp(cleaned)
        
        # Keep tokens that are not stopwords and are alphabetic
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        
        return " ".join(tokens)

    def batch_process(self, texts):
        """Processes a list of strings efficiently."""
        return [self.preprocess(t) for t in texts]
