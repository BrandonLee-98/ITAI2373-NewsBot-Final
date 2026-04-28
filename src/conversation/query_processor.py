import re
from src import (
    NewsClassifier, SentimentAnalyzer, IntelligentSummarizer, 
    SemanticSearchEngine, NewsLanguageDetector, NewsTranslator
)

class QueryProcessor:
    def __init__(self, classifier_path=None):
        self.classifier = NewsClassifier(classifier_path)
        self.sentiment = SentimentAnalyzer()
        self.summarizer = IntelligentSummarizer()
        self.search_engine = SemanticSearchEngine()
        self.detector = NewsLanguageDetector()
        self.translator = NewsTranslator()
        
    def detect_intent(self, user_query):
        """Identifies user goals, now including translation."""
        query = user_query.lower()
        
        if any(word in query for word in ["translate", "language", "spanish", "french"]):
            return "translate"
        elif any(word in query for word in ["summarize", "summary", "brief"]):
            return "summarize"
        elif any(word in query for word in ["find", "search", "looking for"]):
            return "search"
        elif any(word in query for word in ["analyze", "sentiment", "mood"]):
            return "analyze"
        return "general_query"

    def process(self, user_query, article_text=None, article_db=None):
        intent = self.detect_intent(user_query)
        
        # Translation Workflow
        if intent == "translate" and article_text:
            lang = self.detector.identify_language(article_text)
            if lang == 'en':
                return "This article is already in English."
            translation = self.translator.translate_to_english(article_text)
            return (f"Detected Language: {lang}\n"
                    f"Translation: {translation['translated']}")

        # Summarization Workflow
        elif intent == "summarize" and article_text:
            return f"Summary: {self.summarizer.summarize(article_text)}"
            
        # Analysis Workflow
        elif intent == "analyze" and article_text:
            metrics = self.sentiment.get_sentiment_metrics(article_text)
            return f"The tone is {metrics['label']} (Polarity: {metrics['polarity']})."

        return "I'm ready to help! Try 'summarize this' or 'translate this'."
