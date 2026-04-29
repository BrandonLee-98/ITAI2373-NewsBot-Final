from deep_translator import GoogleTranslator

class NewsTranslator:
    def __init__(self):
        # Initializes the translator to automatically detect the source language 
        # and translate it to English
        self.translator = GoogleTranslator(source='auto', target='en')

    def translate(self, text):
        """Translates the provided text into English."""
        try:
            return self.translator.translate(text)
        except Exception as e:
            return f"Translation error: {str(e)}"
