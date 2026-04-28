from gensim import corpora, models

class NewsTopicModeler:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.dictionary = None
        self.lda_model = None

    def train(self, processed_docs):
        """Trains the LDA model on the provided documents."""
        self.dictionary = corpora.Dictionary(processed_docs)
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        self.lda_model = models.LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=10
        )

    def get_topics(self):
        """Returns the top words for each discovered topic."""
        if not self.lda_model: return "Model not trained."
        return self.lda_model.print_topics(num_words=5)
