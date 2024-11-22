# **Week 10: Supervised Learning for Sentiment Analysis**

## **Objectives:**

Week 10 focuses on applying supervised learning techniques for sentiment analysis, a critical task in natural language processing (NLP). By training models on labeled datasets, we aim to classify text data into sentiment categories such as positive, negative, or neutral.

- Understand the role of supervised learning in sentiment analysis.
- Explore the use of word embeddings as input features for classification models.
- Train and evaluate sentiment classification models using metrics like accuracy and F1-score.

## **Code Examples**

### Example 1 - Sentiment Analysis with Logistic Regression:
In this example, we train a Logistic Regression model for sentiment classification. The model is trained on a dataset of text reviews, with each review labeled as positive or negative.

**Highlights**:
- **Text Representation**: Input text is converted into numerical representations using TF-IDF vectors or word embeddings.
- **Training Process**: Logistic Regression is trained using labeled data to classify the sentiment of text reviews.
- **Evaluation**: Model performance is evaluated using accuracy and F1-score.

**Applications**:
- Customer feedback analysis.
- Social media sentiment tracking.

### Example 2 - Sentiment Analysis with Neural Networks:
This example demonstrates the use of a simple feedforward neural network for sentiment classification. The network is trained on word embeddings derived from pre-trained models like GloVe or Word2Vec.

**Highlights**:
- **Input Layer**: Accepts word embeddings as input features.
- **Hidden Layers**: Extract patterns and relationships in the embeddings.
- **Output Layer**: Produces sentiment predictions (e.g., positive, negative).

**Applications**:
- Product review classification.
- Identifying sentiment trends in news articles.

*Note: For full implementation details, refer to the repository.*  

## **Professor Provided Readings:**

- [Affective Text: A Benchmark for Sentiment Analysis](https://aclanthology.org/W04-1013)
- [Deep Learning for Sentiment Analysis: A Survey](https://arxiv.org/abs/1404.2826)

## **Self Researched Readings:**

- [Sentiment Analysis with Deep Learning](https://arxiv.org/abs/1512.07778)
- [Using Word Embeddings for Sentiment Analysis](https://dl.acm.org/doi/10.1145/3178876.3186007)
