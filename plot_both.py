import matplotlib.pyplot as plt


def plot(sentiment_prediction, entity_prediction):
    # Create a new figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Add the sentiment plot to the second subplot
    ax2.bar(['Negative', 'Positive'], [1 - sentiment_prediction[0],
                                       sentiment_prediction[0]])
    ax2.set_xlabel("Sentiment")
    ax2.set_ylabel("Probability")

    # Add the entity plot to the first subplot
    entity_prediction = [int(p * 100) for p in entity_prediction]
    ax1.pie(entity_prediction, labels=['Movie', 'Place', 'App', 'Product'], autopct="%0.2f%%")
    ax1.legend()

    return fig
