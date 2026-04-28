from .translator import NewsTranslator
from .language_detector import NewsLanguageDetector

class CrossLingualAnalyzer:
    """
    Compares coverage and sentiment across different language sources.
    Fulfills 'Cross-Language Analysis' and 'Cultural Context' requirements.
    """
    
    def __init__(self, sentiment_analyzer):
        self.translator = NewsTranslator()
        self.detector = NewsLanguageDetector()
        self.sentiment = sentiment_analyzer

    def compare_perspectives(self, articles):
        """
        Takes a list of articles in various languages and compares their sentiment.
        'articles' should be a list of dictionaries: [{'text': '...', 'lang': '...'}, ...]
        """
        comparison_results = []
        
        for item in articles:
            text = item['text']
            # Detect language if not provided
            lang = item.get('lang') or self.detector.identify_language(text)
            
            # Translate to English for a fair sentiment comparison
            if lang != 'en':
                translation_data = self.translator.translate_to_english(text)
                target_text = translation_data['translated']
            else:
                target_text = text
            
            # Get sentiment on the English version
            sentiment_metrics = self.sentiment.get_sentiment_metrics(target_text)
            
            comparison_results.append({
                "language": lang,
                "original_preview": text[:50] + "...",
                "sentiment_label": sentiment_metrics['label'],
                "polarity": sentiment_metrics['polarity']
            })
            
        return comparison_results