from transformers import pipeline

class Summarizer:
    def __init__(self):
        # Loads a dedicated Abstractive Summarization model
        # distilbart-cnn is optimized for news articles and runs efficiently
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    def summarize(self, text):
        # Transformers need a minimum amount of text to generate a good summary
        if not text or len(text.split()) < 30:
            return "Article too short for a meaningful abstractive summary. Please provide a longer text."
            
        try:
            # Generate the summary
            # max_length and min_length control how concise the output is
            result = self.summarizer(text, max_length=60, min_length=20, do_sample=False)
            
            # Extract and clean up the generated text
            summary_text = result[0]['summary_text']
            return summary_text.strip()
            
        except Exception as e:
            print(f"Abstractive Summarizer Error: {e}")
            return "Could not generate summary at this time. Please check the server logs."
