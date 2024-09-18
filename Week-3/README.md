# **Week 3: Language modeling and N-grams**


## **Objectives:**

Week 3 focuses on Language Modeling and N-grams are two fundamental concepts in the fields of Machine Learning (ML) and Natural Language Processing (NLP).
To gain some hands-on experience, we implemented 2 code examples, briefy explained below:

### Example 1: Simple Language Model and N-Grams

In this example, we learned about language modeling and N-grams by creating a simple language model using a short text. The text we have chosen is: "A language model is a type of probabilistic model that assigns probabilities to sequences of words. It is used in various natural language processing tasks."

We used N-grams to break this text into smaller chunks. For instance, 2-grams (bigrams) will group two words together, like "A language," "language model," and so on. The idea is to find patterns in the text and understand how frequently certain word sequences occur.

Once we have our N-grams, we built the language model. It will look at each sequence of words and remember which word tends to follow others. For example, after "A language," the word "model" often appears.

Finally, we used this language model to generate new text. We'll start with a random set of words from the original text and use the model to predict the next word. Then, we'll keep adding words based on the probabilities learned from the N-grams.

The result will is new text that looks similar to the original one but may have some variations. It's like a computer trying to write sentences that sound like a human!

### Example 2: Complex Language Model and Brown Corpus

In this example, we dove into a more advanced language model using a larger collection of text called the Brown corpus. This corpus contains text from various sources, such as news articles, fiction stories, and more.

First, downloaded and loaded the Brown corpus using the Natural Language Toolkit (NLTK) library. The NLTK library helps us work with human language data in Python.

Next, instead of using bigrams as before, we used trigrams (3-grams). Trigrams group three words together, allowing us to capture more complex patterns in the text.

After creating the trigrams, we built the language model, just like we did in the previous example. This time, the model analyzed larger chunks of text and learn more about how words connect to each other in different genres.

Finally, we used this more powerful language model to generate a longer piece of text, trying to mimic the writing style found in the Brown corpus.

The result is an extended text that sounds like it was written by an author using similar writing styles to those found in the Brown corpus. This is the kind of thing used in more advanced natural language processing tasks, like chatbots and text generation applications.


## **Professor Provided Readings:**

[Beyond n-grams: Can linguistic sophistication improve language modeling?](https://aclanthology.org/P98-1028.pdf)

[Show some love to your n-grams: A bit of progress and stronger n-gram language modeling baselines](https://api.repository.cam.ac.uk/server/api/core/bitstreams/83d21f26-066b-4894-915b-63c7749b8a3f/content)

[Building Wikipedia N-grams with Apache Spark](https://www.researchgate.net/profile/Jorge-Fonseca-10/publication/361805716_Building_Wikipedia_N-grams_with_Apache_Spark/links/63146b815eed5e4bd1468051/Building-Wikipedia-N-grams-with-Apache-Spark.pdf)


## **Self-Researched Readings:**  

[The Role of n-gram Smoothing in the Age of Neural Networks](https://github.com/rycolab/ngram_regularizers)
