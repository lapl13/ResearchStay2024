# **Week 7: Word Embeddings in NLP Applications**


## **Objectives:**

Week 7 focuses on word embeddings, a somewhat recent key nreakthrough in the field of Natural Language Processing. The basic idea is that a word embedding is  a conversion of a word or phrease to a number vector, which can facilitate semantic and syntactic realtionships and context establishing in a corpus.
 
## Code Examples

### Example 1 - Simple Word Embeddings:

In this example, we introduce the concept of Word Embeddings using the Word2Vec model with Gensim. Word embeddings are dense vector representations that capture semantic relationships between words, enabling NLP models to understand the meaning and context of words.

Data Preparation: We start with a collection of 20 simple sentences, each containing 8 to 10 words. These sentences cover various NLP-related topics, such as Natural Language Processing, Machine Learning, Text Classification, and more.

Building Word2Vec Model: Using Gensim's Word2Vec class, we create a Word2Vec model with the given sentences. We set the vector size to 10, which means each word will be represented as a 10-dimensional vector in the embedding space. A window of 3 indicates that the model will consider a maximum of 3 words before and after the target word while learning the embeddings.

Obtaining Word Embeddings: After training the Word2Vec model, we can access the word embeddings using the model.wv attribute. We demonstrate retrieving the embedding vector for the word 'NLP'.

Finding Similar Words: We also explore the model's ability to find similar words to a given word based on cosine similarity. In this example, we find words similar to 'NLP'.

### Example 2 - Complex Word Embeddings (Pre-trained Model)

In this example, we showcase the use of a pre-trained Word2Vec model from Gensim's gensim.downloader module. Pre-trained models have been trained on large-scale datasets and can capture complex word relationships and meanings.

Loading Pre-trained Model: We load the 'word2vec-google-news-300' model, which contains word embeddings trained on a vast corpus of Google News articles. The model's dimensionality is 300, meaning each word is represented as a 300-dimensional vector.

Word Embeddings and Similar Words: Similar to Example 1, we obtain the word embedding for the word 'cat' from the pre-trained model and find the most similar words based on cosine similarity.


## **Professor Provided Readings:**

[Word Embeddings: A Survey](https://arxiv.org/pdf/1901.09069.pdf)

[A Survey of Word Embeddings Evaluation Methods](https://arxiv.org/pdf/1801.09536.pdf)

[Evaluation methods for unsupervised word embeddings](https://aclanthology.org/D15-1036.pdf)


## **Self Reasearched Reading:**

[Word Embeddings for Sentiment Analysis: A Comprehensive Empirical Survey](https://arxiv.org/abs/1902.00753)
