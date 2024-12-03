import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
import psutil
import os
import time

# Parameters
MAX_SEQ_LENGTH = 50
BATCH_SIZE = 128
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
EPOCHS = 25

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=';', header=None, names=['text', 'emotion'])
    return df['text'].values, df['emotion'].values

train_texts, train_labels = load_data('data/train.txt')
test_texts, test_labels = load_data('data/test.txt')

# Build vocabulary
word_counter = Counter()
for text in train_texts:
    word_counter.update(text.split())
vocab = ['<PAD>', '<UNK>'] + [word for word, freq in word_counter.items() if freq > 1]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

# Tokenize and encode labels
def tokenize(text):
    return [word_to_idx.get(word, word_to_idx['<UNK>']) for word in text.split()]

def encode_labels(labels):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels), label_encoder

encoded_train_labels, label_encoder = encode_labels(train_labels)
encoded_test_labels, _ = encode_labels(test_labels)

# Dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = [torch.tensor(tokenize(text)) for text in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Padding function
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=word_to_idx['<PAD>'])
    return texts, torch.tensor(labels)

# Data Loaders
train_dataset = EmotionDataset(train_texts, encoded_train_labels, word_to_idx)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

test_dataset = EmotionDataset(test_texts, encoded_test_labels, word_to_idx)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# LSTM Model
class LSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(LSTMEmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word_to_idx['<PAD>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)

# Model initialization
model = LSTMEmotionClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# Training loop with timing
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
training_time = time.time() - start_time

# Evaluate the model with additional metrics
def evaluate_model_with_metrics(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in data_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    return all_preds, all_labels

# Test the model
test_preds, test_labels = evaluate_model_with_metrics(model, test_loader)
report = classification_report(test_labels, test_preds, target_names=label_encoder.classes_, zero_division=0, output_dict=True)

# Display metrics
print(f"Training Time: {training_time:.2f} seconds")
print(f"Test Accuracy: {report['accuracy'] * 100:.2f}%")
for emotion, metrics in report.items():
    if emotion in label_encoder.classes_:  # Filter only emotion classes
        print(f"{emotion.capitalize()}: F1-Score = {metrics['f1-score']:.2f}, Support = {metrics['support']}")

# Profiling resource usage
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
model_size = os.path.getsize('lstm_emotion_classifier_model.pth') / (1024 * 1024)  # Convert to MB
print(f"Memory Usage: {memory_usage:.2f} MB")
print(f"Model Size: {model_size:.2f} MB")

# Save the model to disk
torch.save(model.state_dict(), 'lstm_emotion_classifier_model.pth')
