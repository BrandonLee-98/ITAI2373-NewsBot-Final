''' ✨ Semantic Embeddings
To further enhance our feature set and address the initial request for "embeddings," we can generate semantic embeddings for our news articles. Semantic embeddings capture the meaning of text in a dense vector space, allowing models to understand contextual similarities between articles.
We will use a pre-trained sentence-transformers model, specifically all-MiniLM-L6-v2, which is efficient and provides good general-purpose embeddings. '''

# Install the sentence-transformers library
!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer

print("Loading Sentence-Transformer model...")
# Load a pre-trained sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Sentence-Transformer model loaded!")

def generate_embeddings(texts):
    """
    Generates semantic embeddings for a list of texts.
    """
    # Ensure texts are strings, handle potential NaNs
    texts = [str(text) if pd.notna(text) else "" for text in texts]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    return embeddings

# Generate embeddings for the full_text_processed column
print("Generating semantic embeddings for articles...")
article_embeddings = generate_embeddings(df['full_text_processed'].tolist())

# Add embeddings to the DataFrame (optional, could also be kept separate)
df['semantic_embeddings'] = list(article_embeddings)

print(f"✅ Embeddings generated! Shape: {article_embeddings.shape}")
print("Sample embedding for the first article:")
print(df['semantic_embeddings'].iloc[0][:10]) # Display first 10 dimensions of the first embedding


