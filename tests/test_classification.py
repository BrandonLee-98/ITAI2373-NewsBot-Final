import unittest
from src.analysis.classifier import NewsClassifier

class TestClassification(unittest.TestCase):
    def setUp(self):
        self.classifier = NewsClassifier()

    def test_output_format(self):
        """Ensures the classifier returns the required dictionary structure."""
        text = "The Federal Reserve adjusted interest rates today."
        result = self.classifier.predict(text)
        self.assertIn('label', result)
        self.assertIn('confidence', result)

    def test_score_range(self):
        """Ensures confidence scores are within a valid 0-100 range."""
        text = "OpenAI announced a new reasoning model."
        result = self.classifier.predict(text)
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 100)

if __name__ == '__main__':
    unittest.main()
