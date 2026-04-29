class TopicModeler:
    def get_article_topics(self, text):
        """Extracts basic themes from the article text."""
        if not text:
            return ["General"]
            
        text_lower = text.lower()
        topics = []
        
        # Simple heuristic matching to avoid heavy library crashes
        if any(word in text_lower for word in ["court", "law", "amendment", "liability"]):
            topics.append("Legal Framework")
        if any(word in text_lower for word in ["tech", "ai", "software", "openai"]):
            topics.append("Artificial Intelligence")
        if any(word in text_lower for word in ["market", "fed", "stock", "inflation"]):
            topics.append("Economic Trends")
            
        return topics if topics else ["General News"]

    def get_topic_words(self, topic_id=0):
        """Fallback method required by the original app.py architecture."""
        return ["news", "update", "report"]
