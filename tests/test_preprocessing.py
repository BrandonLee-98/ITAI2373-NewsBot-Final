import unittest
from src.data_processing.text_preprocessor import TextPreprocessor

class TestTextPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_cleaning_logic(self):
        """Verifies that HTML tags and special characters are removed."""
        raw_html = "<h1>Breaking News!</h1> Check out this <a href='#'>link</a>."
        clean_text = self.preprocessor.clean(raw_html)
        self.assertNotIn("<h1>", clean_text)
        self.assertNotIn("</a>", clean_text)

    def test_case_normalization(self):
        """Verifies text is converted to lowercase for model consistency."""
        upper_text = "HOUSTON CITY COLLEGE"
        clean_text = self.preprocessor.clean(upper_text)
        self.assertEqual(clean_text, "houston city college")

if __name__ == '__main__':
    unittest.main()
