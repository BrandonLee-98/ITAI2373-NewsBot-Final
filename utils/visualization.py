# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_sentiment_trend(dates, scores, output_path='results/visualizations/sentiment_trend.png'):
    """
    Generates a line plot showing sentiment evolution over time.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=dates, y=scores, marker='o', color='#2563eb')
    plt.title('Sentiment Evolution Analysis')
    plt.xlabel('Date')
    plt.ylabel('Polarity Score (-1 to 1)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Sentiment trend saved to {output_path}")

def plot_topic_distribution(topic_weights, topic_labels, output_path='results/visualizations/topic_dist.png'):
    """
    Generates a bar chart showing the distribution of topics in a corpus.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=topic_weights, y=topic_labels, palette='viridis')
    plt.title('Discovered Topic Distribution (LDA)')
    plt.xlabel('Weight/Prominence')
    plt.ylabel('Topic Category')
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Topic distribution chart saved to {output_path}")
