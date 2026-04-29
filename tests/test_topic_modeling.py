import unittest
from src.analysis.topic_modeler import TopicModeler

class TestTopicModeling(unittest.TestCase):
    def setUp(self):
        # Initialize with a small number of topics for testing [cite: 129]
        self.modeler = TopicModeler(n_topics=2)
        self.sample_docs = [
            "AI and machine learning are transforming the tech industry.",
            "The stock market saw significant gains in the finance sector.",
            "New robotics models are improving manufacturing efficiency.",
            "Federal interest rates remain a concern for global investors."
        ]

    def test_fit_transform(self):
        """Verifies the model can process documents and generate a distribution[cite: 133]."""
        try:
            self.modeler.fit_transform(self.sample_docs)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"TopicModeler fit_transform failed: {e}")

    def test_topic_words_extraction(self):
        """Verifies that the model returns top words for a discovered topic[cite: 136]."""
        self.modeler.fit_transform(self.sample_docs)
        words = self.modeler.get_topic_words(topic_id=0, n_words=3)
        self.assertEqual(len(words), 3)
        self.assertIsInstance(words[0], str)

if __name__ == '__main__':
    unittest.main()
