import matplotlib.pyplot as plt

def plot_sentiment_trend(data):
    """Fulfills Sentiment Evolution tracking requirement """
    plt.plot(data['date'], data['polarity'])
    plt.show()
