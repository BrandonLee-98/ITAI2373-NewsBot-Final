import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

class SemanticSearchEngine:
    """
    Advanced semantic search using sentence embeddings.
    Allows for 'meaning-based' retrieval rather than just keyword matching.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        'all-MiniLM-L6-v2' is a fast, high-quality model perfect for news search.
        """
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus_articles = None

    def fit(self, articles):
        """
        Encodes the entire news database into a vector space.
        """
        self.corpus_articles = articles
        print(f"Encoding {len(articles)} articles into semantic vectors...")
        self.corpus_embeddings = self.model.encode(articles, convert_to_tensor=True)
        print("Encoding complete.")

    def search(self, query, top_k=3):
        """
        Finds the most semantically similar articles to a user query.
        Fulfills the 'Semantic Search' requirement for Module B.
        """
        if self.corpus_embeddings is None:
            return "Error: Search engine not fitted with data."

        # Encode the user query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity between query and all articles
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        # Get the top K results
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus_articles)))

        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            results.append({
                "article": self.corpus_articles[idx],
                "score": round(float(score), 4)
            })
        
        return results

if __name__ == "__main__":
    # Test Data
    news_db = [
        "The economy is facing a significant downturn due to inflation.",
        "A new iPhone model was released with a faster processor.",
        "Major league baseball teams are preparing for the spring season.",
        "The central bank announced new interest rate policies.",
        "Tech giants are investing billions into generative AI research."
    ]

    search_engine = SemanticSearchEngine()
    search_engine.fit(news_db)

    # Test Query
    query = "financial crisis and rising prices"
    print(f"\nQuery: {query}")
    
    hits = search_engine.search(query)
    for hit in hits:
        print(f"Score: {hit['score']} | Content: {hit['article']}")