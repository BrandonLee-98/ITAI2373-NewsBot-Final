import spacy

class EntityRelationshipMapper:
    def __init__(self):
        # Load the English NLP model safely
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        if not text:
            return []
            
        doc = self.nlp(text)
        entities = []
        
        # Only pull high-value entity types
        valid_labels = ["PERSON", "ORG", "GPE", "DATE"]
        for ent in doc.ents:
            if ent.label_ in valid_labels:
                entities.append({"text": ent.text, "label": ent.label_})
                
        # Remove duplicates while preserving order
        return [dict(t) for t in {tuple(d.items()) for d in entities}]
