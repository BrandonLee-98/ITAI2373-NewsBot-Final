from textblob import TextBlob

class SentimentAnalyzer:
    def get_sentiment_metrics(self, text):
        if not text:
            return {"score": 0, "label": "Neutral"}
            
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Using the calibrated 0.05 threshold from your Technical Report
        if polarity > 0.05:
            label = "Positive"
        elif polarity < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        return {"score": round(polarity, 2), "label": label}
