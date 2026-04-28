import spacy

class NERExtractor:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def extract_entities(self, text):
        """Focusing on PERSON, ORG (Organizations), GPE (Locations), and DATE."""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]:
                entities.append({"text": ent.text, "label": ent.label_})
        return entities
