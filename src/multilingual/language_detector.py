from langdetect import detect, detect_langs

class NewsLanguageDetector:
    """
    Automatic language identification for news articles.
    Provides confidence scoring for the detected language.
    """
    
    def __init__(self):
        pass

    def identify_language(self, text):
        """
        Detects the primary language of the text.
        Fulfills 'Language Detection' requirement.
        """
        if not text or len(text.strip()) < 5:
            return "unknown"
        try:
            return detect(text)
        except:
            return "unknown"

    def get_confidence_scores(self, text):
        """
        Returns all possible languages with their probability scores.
        """
        try:
            # Returns a list of Language objects (e.g., [en:0.99, es:0.01])
            return detect_langs(text)
        except:
            return []

if __name__ == "__main__":
    test_text = "El mercado de valores cerró al alza hoy."
    detector = NewsLanguageDetector()
    print(f"Detected Language: {detector.identify_language(test_text)}")
    print(f"Confidence: {detector.get_confidence_scores(test_text)}")