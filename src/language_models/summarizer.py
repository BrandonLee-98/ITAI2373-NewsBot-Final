from textblob import TextBlob

class Summarizer:
    def summarize(self, text):
        if not text or len(text.split()) < 20:
            return "Article too short to summarize."
            
        # Extractive Summarization: Grab the first and last meaningful sentences
        blob = TextBlob(text)
        sentences = [str(sentence) for sentence in blob.sentences if len(sentence.words) > 5]
        
        if len(sentences) <= 2:
            return text
            
        summary = f"{sentences[0]} ... {sentences[-1]}"
        return summary
