from textblob import TextBlob
import pandas as pd
import numpy as np

class SentimentAnalyzer:
    """
    Advanced sentiment analysis with polarity and subjectivity tracking.
    Designed for NewsBot 2.0's Sentiment Evolution requirement.
    """
    
    def __init__(self):
        # We use TextBlob here for its reliability with news-style text, 
        # but you can easily swap this for VADER or a Transformer model.
        pass

    def get_sentiment_metrics(self, text):
        """
        Returns a detailed dictionary of sentiment scores.
        """
        if not text:
            return {"polarity": 0, "subjectivity": 0, "label": "Neutral"}
            
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Determine the label based on polarity
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
            
        return {
            "polarity": round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
            "label": label
        }

    def track_temporal_evolution(self, dataframe, date_column='date', text_column='text'):
        """
        Calculates sentiment over time.
        Fulfills the 'Sentiment Evolution' requirement for Module A.
        """
        # Ensure the date column is in datetime format
        dataframe[date_column] = pd.to_datetime(dataframe[date_column])
        
        # Apply sentiment analysis to each row
        results = dataframe[text_column].apply(self.get_sentiment_metrics)
        dataframe['polarity'] = [r['polarity'] for r in results]
        
        # Resample by day (or week) and calculate the mean polarity
        evolution = dataframe.set_index(date_column).resample('D')['polarity'].mean()
        
        return evolution

if __name__ == "__main__":
    # Test Data
    data = {
        'date': ['2026-04-01', '2026-04-02', '2026-04-03'],
        'text': [
            "The market is booming and everyone is happy!",
            "Stocks took a slight dip today, causing some concern.",
            "A massive crash has devastated local investors."
        ]
    }
    df = pd.DataFrame(data)
    
    analyzer = SentimentAnalyzer()
    evolution = analyzer.track_temporal_evolution(df)
    print("Sentiment Evolution (Mean Polarity per Day):")
    print(evolution)