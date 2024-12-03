import re
from collections import Counter
import time
import psutil
import os
import pickle

emotions = ['love', 'fear', 'sadness', 'surprise', 'joy', 'anger']

stop_words = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
     'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they',
     'them', 'their', 'theirs', 'themselves',
     'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
     'be', 'been', 'being', 'have', 'has', 'had',
     'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
     'while', 'of', 'at', 'by', 'for', 'with', 'about',
     'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
     'in', 'out', 'on', 'off', 'over', 'under',
     'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
     'few', 'more', 'most', 'other', 'some',
     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
     'don', "don't", 'should', "should've",
     'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
     'doesn', "doesn't", 'hadn', "hadn't",
     'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
     "needn't", 'shan', "shan't", 'shouldn',
     "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'feeling', 'feel',
     'really', 'im', 'like',
     'know', 'get', 'ive', "im'", 'stil', 'even', 'time', 'want', 'one', 'cant', 'think', 'go', 'much', 'never', 'day',
     'back', 'see', 'still', 'make', 'thing',
     'would', "would'", "could'", 'little', ])


def create_lexicon(file_path, counter_most_common):
    emotion_counters = {emotion: Counter() for emotion in emotions}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, emotion = line.strip().split(';')
            if emotion in emotions:
                words = [word for word in re.findall(r'\w+', text.lower()) if word not in stop_words]
                emotion_counters[emotion].update(words)

    emotion_lexicon = {emotion: [word for word, _ in counter.most_common(counter_most_common)] for emotion, counter in
                       emotion_counters.items()}

    # Save lexicon to calculate model size
    with open("emotion_lexicon.pkl", "wb") as f:
        pickle.dump(emotion_lexicon, f)

    return emotion_lexicon


def predict(file_path, lexicon):
    correct_predictions = 0
    total_predictions = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, actual_emotion = line.strip().split(';')
            words = set(re.findall(r'\w+', text.lower()))

            emotion_scores = {emotion: 0 for emotion in lexicon}

            for word in words:
                for emotion, emotion_words in lexicon.items():
                    if word in emotion_words:
                        emotion_scores[emotion] += 1

            predicted_emotion = max(emotion_scores, key=emotion_scores.get)

            if predicted_emotion == actual_emotion:
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


def calculate_model_size():
    # Calculate model size
    model_size = os.path.getsize("emotion_lexicon.pkl") / (1024 * 1024)  # Size in MB
    return model_size


if __name__ == '__main__':
    train_file_path = 'data/train.txt'
    test_file_path = 'data/val.txt'

    # Hyperparameters:
    increment_step = 100
    max_word_count = 350

    # Profiling metrics
    memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory before processing in MB

    start_time = time.time()  # Start timer

    for counter_most_common in range(5, max_word_count, increment_step):
        lexicon = create_lexicon(train_file_path, counter_most_common)
        accuracy = predict(test_file_path, lexicon)
        model_size = calculate_model_size()

        # Profiling current memory usage
        memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory after processing in MB

        # Time elapsed
        elapsed_time = time.time() - start_time

        print(f"Counter Most Common: {counter_most_common}, "
              f"Accuracy: {accuracy * 100:.2f}%, "
              f"Model Size: {model_size:.2f} MB, "
              f"Memory Usage: {memory_after - memory_before:.2f} MB, "
              f"Elapsed Time: {elapsed_time:.2f} seconds")

    print("End")
