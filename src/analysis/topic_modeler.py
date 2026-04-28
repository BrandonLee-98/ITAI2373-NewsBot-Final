from gensim import corpora
from gensim.models.ldamodel import LdaModel
from src.data_processing.text_preprocessor import TextPreprocessor

class TopicDiscoveryEngine:
    """
    Advanced topic modeling for discovering hidden themes and trends.
    Implements LDA for content discovery and clustering.
    """
    
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.preprocessor = TextPreprocessor()
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

    def prepare_corpus(self, documents):
        """
        Tokenizes text and creates the dictionary and bag-of-words corpus.
        """
        # Clean and tokenize the documents
        processed_docs = [self.preprocessor.process(doc).split() for doc in documents]
        
        # Create Dictionary and Corpus
        self.dictionary = corpora.Dictionary(processed_docs)
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_docs]
        
        return self.corpus

    def fit(self, documents):
        """
        Trains the LDA model on the provided documents[cite: 1550].
        """
        corpus = self.prepare_corpus(documents)
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            random_state=42,
            passes=10,
            alpha='auto'
        )
        print(f"LDA Model trained with {self.n_topics} topics.")

    def get_topics(self, n_words=10):
        """
        Returns the top words for each discovered topic[cite: 1553].
        """
        topics = self.lda_model.print_topics(num_words=n_words)
        return topics

    def get_document_topics(self, text):
        """
        Predicts the topic distribution for a new, single article.
        Fulfills the 'Content Clustering' requirement.
        """
        processed = self.preprocessor.process(text).split()
        bow = self.dictionary.doc2bow(processed)
        return self.lda_model.get_document_topics(bow)

if __name__ == "__main__":
    # Test dataset
    news_samples = [
        "The central bank raised interest rates to combat inflation and help the economy.",
        "New software update for smartphones improves battery life and camera performance.",
        "The championship game ended in a tie after a thrilling overtime period.",
        "Economists predict a slow recovery for the financial market this quarter.",
        "The new AI model can generate realistic images from text prompts."
    ]
    
    engine = TopicDiscoveryEngine(n_topics=3)
    engine.fit(news_samples)
    
    print("\n--- Discovered Topics ---")
    for topic in engine.get_topics():
        print(topic)