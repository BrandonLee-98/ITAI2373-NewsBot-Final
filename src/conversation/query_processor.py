from transformers import pipeline

class QueryProcessor:
    def __init__(self):
        # Loads a lightweight, lightning-fast Question Answering model
        self.qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

    def process(self, query, context=""):
        if not context:
            return "Please paste an article above so I have some context to read!"
        
        try:
            # The model scans the context and extracts the exact answer
            result = self.qa_model(question=query, context=context)
            
            # Return the answer if the model is reasonably confident
            if result['score'] > 0.05:
                # Capitalize the first letter for clean formatting
                answer = result['answer'].capitalize()
                return f"According to the text: {answer}."
            else:
                return "I couldn't find a clear answer to that specific question in the article."
                
        except Exception as e:
            print(f"QA Model Error: {e}")
            return "I ran into a minor issue analyzing that question. Try asking it a different way!"
