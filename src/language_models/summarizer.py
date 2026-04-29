from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    def __init__(self):
        # We bypass the 'pipeline' wrapper and load the exact model and tokenizer explicitly.
        # This prevents environment registry errors and gives us more control.
        model_name = "sshleifer/distilbart-cnn-12-6"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text):
        if not text or len(text.split()) < 30:
            return "Article too short for a meaningful abstractive summary. Please provide a longer text."
            
        try:
            # 1. Tokenize the input text (convert words to numbers the model understands)
            # truncation=True ensures we don't crash if the article is incredibly long
            inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            
            # 2. Generate the summary using beam search for higher quality
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                max_length=60, 
                min_length=20, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            
            # 3. Decode the output numbers back into human-readable text
            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary_text.strip()
            
        except Exception as e:
            print(f"Abstractive Summarizer Error: {e}")
            return "Could not generate summary at this time. Please check the server logs."
