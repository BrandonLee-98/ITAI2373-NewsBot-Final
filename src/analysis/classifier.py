from transformers import pipeline

class NewsClassifier:
    def __init__(self):
        # We load a Zero-Shot Classification model.
        # This is a powerful, lightweight model that satisfies the requirement 
        # for multi-level categorization and confidence scoring without needing a custom trained dataset.
        try:
            self.classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
        except Exception as e:
            print(f"Classifier Load Error: {e}")
            self.classifier = None
            
        # Define our multi-level categories (you can add or change these anytime!)
        self.categories = [
            "Legal & Politics", 
            "Technology & Artificial Intelligence", 
            "Finance & Markets", 
            "Global Health & Science", 
            "Entertainment & Culture"
        ]

    def predict(self, text):
        if not text or len(text.strip()) < 10:
            return {"label": "Uncategorized", "confidence": 0.0}
            
        if not self.classifier:
            return {"label": "System Loading...", "confidence": 0.0}

        try:
            # The model scores the text against all candidate labels
            result = self.classifier(text, candidate_labels=self.categories, multi_label=False)
            
            # Extract the top matching category and its probability (confidence score)
            top_category = result['labels'][0]
            confidence_score = result['scores'][0]
            
            # Return a dictionary containing both the label and the confidence percentage
            return {
                "label": top_category, 
                "confidence": round(confidence_score * 100, 2)
            }
            
        except Exception as e:
            print(f"Classification Prediction Error: {e}")
            return {"label": "Classification Error", "confidence": 0.0}
