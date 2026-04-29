class NewsClassifier:
    def predict(self, text):
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["court", "law", "amendment", "liability", "supreme"]):
            return {"label": "Legal & Politics"}
        elif any(word in text_lower for word in ["tech", "ai", "software", "openai", "infrastructure"]):
            return {"label": "Technology"}
        elif any(word in text_lower for word in ["market", "fed", "stock", "inflation", "finance"]):
            return {"label": "Finance"}
            
        return {"label": "General News"}
