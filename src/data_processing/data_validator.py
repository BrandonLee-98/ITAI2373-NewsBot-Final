# This section contains code snippets that perform data quality checks, such as examining dataset overview, category distribution, and basic preprocessing validation.
# Basic dataset exploration (from cell ZCaEOOJKowrE)
print("📊 DATASET OVERVIEW")
print("=" * 50)
print(f"Total articles: {len(df)}")
print(f"Unique categories: {df['category'].nunique()}")
print(f"Categories: {df['category'].unique().tolist()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

print("\n📈 CATEGORY DISTRIBUTION")
print("=" * 50)
category_counts = df['category'].value_counts()
print(category_counts)

# Visualize category distribution
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='category', order=category_counts.index)
plt.title('Distribution of News Categories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Analyze preprocessing results (student task from cell N7qUwHPSowrE)
# Calculate average text length before and after
print("\n📝 PREPROCESSING VALIDATION")
print("=" * 60)
print(f"Average original full text length: {df['full_text'].str.len().mean():.2f}")
print(f"Average processed full text length: {df['full_text_processed'].str.split().str.len().apply(lambda x: sum(len(word) for word in x)).mean():.2f}")

# Count unique words before and after
from collections import Counter

original_words = Counter(' '.join(df['full_text'].dropna()).split())
processed_words = Counter(' '.join(df['full_text_processed'].dropna()).split())

print(f"Unique words (original): {len(original_words)}")
print(f"Unique words (processed): {len(processed_words)}")

# Identify the most common words after preprocessing
print("\nMost common words after preprocessing:")
for word, count in processed_words.most_common(10):
    print(f"  {word}: {count}")
