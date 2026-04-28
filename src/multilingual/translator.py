from googletrans import Translator

class NewsTranslator:
    """
    Handles translation workflows for cross-language content access.
    Fulfills 'Translation Integration' requirement.
    """
    
    def __init__(self):
        self.translator = Translator()

    def translate_to_english(self, text):
        """
        Translates source text into English.
        """
        try:
            result = self.translator.translate(text, dest='en')
            return {
                "original": text,
                "translated": result.text,
                "source_lang": result.src
            }
        except Exception as e:
            return {"error": f"Translation failed: {str(e)}"}

if __name__ == "__main__":
    # Test translation from Spanish to English
    translator = NewsTranslator()
    sample = "La inteligencia artificial está transformando el periodismo."
    print(translator.translate_to_english(sample))