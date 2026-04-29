import re
from src import (
    NewsClassifier, SentimentAnalyzer, Summarizer, 
    SemanticSearchEngine, NewsLanguageDetector, NewsTranslator
)

class QueryProcessor:
    def __init__(self, classifier_path=None):
        self.classifier = NewsClassifier(classifier_path)
        self.sentiment = SentimentAnalyzer()
        self.summarizer = Summarizer()
        self.search_engine = SemanticSearchEngine()
        self.detector = NewsLanguageDetector()
        self.translator = NewsTranslator()
        
    def detect_intent(self, user_query):
        """Identifies user goals, now including translation."""
        query = user_query.lower()
        
        if any(word in query for word in ["translate", "language", "spanish"]):
            return "translate"
        elif any(word in query for word in ["summarize", "summary"]):
            return "summarize"
        elif any(word in query for word in ["analyze", "sentiment"]):
            return "analyze"
        return "general_query"

    def process(self, user_query, article_text=None, article_db=None):
        intent = self.detect_intent(user_query)
        
        if intent == "translate" and article_text:
            lang = self.detector.identify_language(article_text)
            if lang == 'en': return "Already in English."
            translation = self.translator.translate_to_english(article_text)
            return f"Language: {lang}\nTranslation: {translation['translated']}"

        elif intent == "summarize" and article_text:
            return self.summarizer.summarize(article_text)
            
        elif intent == "analyze" and article_text:
            metrics = self.sentiment.get_sentiment_metrics(article_text)
            return f"Tone: {metrics['label']} (Polarity: {metrics['polarity']})"

        return "How can I help with your news analysis today?"
