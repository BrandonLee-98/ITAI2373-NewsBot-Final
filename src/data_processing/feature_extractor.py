# This section contains the code for TF-IDF feature extraction and the combination of TF-IDF with sentiment, text length, and POS features into a comprehensive feature matrix.
# Create TF-IDF vectorizer
# 💡 TIP: Experiment with different parameters:
# - max_features: limit vocabulary size
# - ngram_range: include phrases (1,1) for words, (1,2) for words+bigrams
# - min_df: ignore terms that appear in less than min_df documents
# - max_df: ignore terms that appear in more than max_df fraction of documents

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit vocabulary for computational efficiency
    ngram_range=(1, 2),  # Include unigrams and bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.8  # Ignore terms that appear in more than 80% of documents
)

# Fit and transform the processed text (assuming 'df' DataFrame exists with 'full_text_processed')
print("🔢 Creating TF-IDF features...")
tfidf_matrix = tfidf_vectorizer.fit_transform(df['full_text_processed'])
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"✅ TF-IDF matrix created!")
print(f"📊 Shape: {tfidf_matrix.shape}")
print(f"📝 Vocabulary size: {len(feature_names)}")
print(f"🔢 Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")

# Prepare features for classification
print("🔧 Preparing combined features...")

# TF-IDF features
X_tfidf = tfidf_matrix.toarray()  # Convert sparse matrix to dense for concatenation

# Sentiment features (assuming 'sentiment_df' DataFrame exists)
sentiment_features = sentiment_df[['full_sentiment', 'pos_score', 'neu_score', 'neg_score']].values

# Text length features (assuming 'df' DataFrame exists)
length_features = np.array([
    df['full_text'].str.len(),  # Character length
    df['full_text'].str.split().str.len(),  # Word count
    df['headline'].str.len(),  # Title length
]).T

# POS features (assuming 'pos_df' DataFrame exists)
# Ensure pos_df is aligned with df based on article_id
pos_features_aligned = df[['article_id']].merge(pos_df, on='article_id', how='left').drop(columns=['article_id', 'category']).fillna(0).values

# Combine all features
X_combined = np.hstack([
    X_tfidf,
    sentiment_features,
    length_features,
    pos_features_aligned
])

print(f"✅ Combined feature matrix prepared!")
print(f"📊 Combined feature matrix shape: {X_combined.shape}")
