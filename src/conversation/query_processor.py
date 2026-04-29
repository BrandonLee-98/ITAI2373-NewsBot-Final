class QueryProcessor:
    def process(self, prompt):
        prompt_lower = prompt.lower()
        
        if "sentiment" in prompt_lower or "tone" in prompt_lower:
            return "Based on my NLP analysis, I calculate sentiment by measuring the polarity of adjectives used in the text. You can see the final verdict in the Intelligence Report above!"
        elif "entity" in prompt_lower or "who" in prompt_lower or "where" in prompt_lower:
            return "I use a spaCy Named Entity Recognition (NER) model to extract key People, Organizations, and Locations from the text."
        elif "summary" in prompt_lower or "main" in prompt_lower:
            return "The article's core themes have been extracted. Please check the Executive Summary box for the direct synthesis."
        else:
            return "I have processed the article! Try asking me about its 'sentiment', the 'entities' involved, or its 'summary'."
