import spacy
from collections import Counter

class EntityRelationshipMapper:
    """
    Advanced NER and relationship extraction system.
    Identifies key players and the connections between them in news text.
    """
    
    def __init__(self, model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            from spacy.cli import download
            download(model)
            self.nlp = spacy.load(model)

    def extract_entities(self, text):
        """
        Extracts named entities and filters for the most relevant news categories.
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            # Focusing on PERSON, ORG (Organizations), GPE (Locations), and DATE [cite: 402]
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'EVENT']:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'explanation': spacy.explain(ent.label_)
                })
        return entities

    def map_relationships(self, text):
        """
        Uses dependency parsing to find relationships (Subject-Verb-Object).
        Fulfills the 'Entity Relationship Mapping' requirement.
        """
        doc = self.nlp(text)
        relationships = []
        
        for sent in doc.sents:
            # Look for triplets: Subject -> Root Verb -> Object
            subjects = [token for token in sent if "subj" in token.dep_]
            for subj in subjects:
                # Find the verb (head) connected to the subject
                verb = subj.head
                # Find the objects connected to that verb
                objects = [token for token in verb.children if "obj" in token.dep_]
                
                for obj in objects:
                    relationships.append({
                        'subject': subj.text,
                        'relationship': verb.lemma_,
                        'object': obj.text
                    })
                    
        return relationships

    def get_knowledge_summary(self, text):
        """
        Provides a combined view of entities and their extracted relationships.
        """
        return {
            "entities": self.extract_entities(text),
            "relationships": self.map_relationships(text)
        }

if __name__ == "__main__":
    # Test sample
    news_text = "Apple CEO Tim Cook announced a new partnership with Goldman Sachs in New York yesterday."
    
    mapper = EntityRelationshipMapper()
    analysis = mapper.get_knowledge_summary(news_text)
    
    print("--- Detected Entities ---")
    for ent in analysis['entities']:
        print(f"{ent['text']} ({ent['label']})")
        
    print("\n--- Extracted Relationships ---")
    for rel in analysis['relationships']:
        print(f"{rel['subject']} --[{rel['relationship']}]--> {rel['object']}")