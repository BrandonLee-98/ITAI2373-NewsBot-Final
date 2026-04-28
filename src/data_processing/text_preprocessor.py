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
            import en_core_web_sm
            self.nlp = en_core_web_sm.load()
            
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Standard cleaning pipeline."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = " ".join(text.split())
        return text

    def preprocess(self, text):
        """Full preprocessing: clean -> tokenize -> lemmatize -> remove stopwords."""
        cleaned = self.clean_text(text)
        doc = self.nlp(cleaned)
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)
