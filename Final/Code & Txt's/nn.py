import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from collections import Counter
import re
import time
import psutil
import os

# Hyperparameters
MAX_SEQ_LENGTH = 65
EPOCHS = 30
EMBED_DIM = 50


# Tokenize and pad/truncate
def tokenize(text, max_length):
    tokens = re.findall(r'\w+', text.lower())
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens + ['<PAD>'] * (max_length - len(tokens))


def load_data(file_path, max_length):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().split(';')
            texts.append(tokenize(text, max_length))
            labels.append(label)
    return texts, labels


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = [[word_to_idx.get(word, word_to_idx['<UNK>']) for word in text] for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])


class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, EMBED_DIM, num_classes):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.fc = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)


# Load and process data
train_texts, train_labels = load_data('data/train.txt', MAX_SEQ_LENGTH)
test_texts, test_labels = load_data('data/val.txt', MAX_SEQ_LENGTH)

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Build vocabulary
all_words = [word for text in train_texts for word in text]
word_counts = Counter(all_words)
vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items() if count > 1]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Model parameters
vocab_size = len(vocab)
num_classes = len(label_encoder.classes_)

# Model, Loss, and Optimizer
model = EmotionClassifier(vocab_size, EMBED_DIM, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data loaders
train_dataset = EmotionDataset(train_texts, train_labels, word_to_idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = EmotionDataset(test_texts, test_labels, word_to_idx)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Measure training time
start_time = time.time()

# Training loop
for epoch in range(EPOCHS):
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")

# Save the model to disk
torch.save(model.state_dict(), 'emotion_classifier_model.pth')

# Load the model from disk
loaded_model = EmotionClassifier(vocab_size, EMBED_DIM, num_classes)
loaded_model.load_state_dict(torch.load('emotion_classifier_model.pth'))

# Evaluate the model
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Classification report
report = classification_report(
    all_labels,
    all_preds,
    target_names=label_encoder.classes_,
    zero_division=0,
    output_dict=True  # Generate the report as a dictionary for custom formatting
)

# Display metrics
print(f"Overall Accuracy: {report['accuracy'] * 100:.2f}%")  # Overall accuracy
for emotion, metrics in report.items():
    if emotion in label_encoder.classes_:  # Filter only emotion classes
        print(
            f"{emotion.capitalize()}: "
            f"F1-Score = {metrics['f1-score']:.2f}, "
            f"Support = {metrics['support']}"
        )

# Memory usage
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
print(f"Memory Usage: {memory_usage:.2f} MB")

# Model size
model_size = os.path.getsize('emotion_classifier_model.pth') / (1024 * 1024)  # Convert to MB
print(f"Model Size: {model_size:.2f} MB")
