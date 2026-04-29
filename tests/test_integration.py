import unittest
import json
from app import app

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        # Set up the Flask test client [cite: 178]
        self.app = app.test_client()
        self.app.testing = True

    def test_analyze_pipeline_flow(self):
        """Tests the full /analyze route from text input to JSON output[cite: 193, 234]."""
        payload = {"text": "El mercado de valores cerró al alza tras el anuncio de OpenAI."}
        response = self.app.post('/analyze', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify all four modules contributed to the result [cite: 18, 234]
        self.assertIn('classification', data)
        self.assertIn('summary', data)
        self.assertIn('language', data) # Module C
        self.assertIn('entities', data)

    def test_query_interface(self):
        """Tests the conversational QA interface[cite: 206]."""
        payload = {
            "query": "What happened to the market?",
            "context": "The stock market closed at an all-time high today."
        }
        response = self.app.post('/query', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)

if __name__ == '__main__':
    unittest.main()
