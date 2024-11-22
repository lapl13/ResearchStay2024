import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import os
import zipfile

# Function to download and join the Go Emotions dataset


def download_goemotions_dataset():
    urls = [
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"
    ]

    data_dir = "data/full_dataset"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for url in urls:
        file_name = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(file_name):
            print(f"Downloading {file_name}...")
            response = requests.get(url)
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"Download complete: {file_name}")

    # Join the CSV files into a single DataFrame
    csv_files = [os.path.join(
        data_dir, f"goemotions_{i}.csv") for i in range(1, 4)]
    df = pd.concat([pd.read_csv(file)
                   for file in csv_files], ignore_index=True)
    return df


# Check if the Go Emotions dataset exists, if not, download and join it
data_path = "data/full_dataset/goemotions_data.csv"
if not os.path.exists(data_path):
    print("Downloading and joining Go Emotions dataset...")
    df = download_goemotions_dataset()
    df.to_csv(data_path, index=False)
    print("Dataset download and join complete.")

# Load Go Emotions dataset
df = pd.read_csv(data_path)

# List of emotions in the dataset
emotion_labels = df.columns[1:].tolist()
num_emotions = len(emotion_labels)

# Preprocess the text and labels
texts = df["text"].tolist()
labels = df[emotion_labels].values.tolist()

# Define the BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=num_emotions)

# Custom Dataset for Go Emotions


class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        inputs = self.tokenizer(text, return_tensors='pt',
                                padding=True, truncation=True)
        return inputs, label


# Create Dataset and DataLoader
dataset = GoEmotionsDataset(texts, labels, tokenizer)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the emotion detection model


class EmotionDetectionModel(nn.Module):
    def __init__(self, num_emotions):
        super(EmotionDetectionModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_emotions)

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        return outputs.logits


# Instantiate the model and move to GPU if available
model = EmotionDetectionModel(num_emotions)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training the model (code omitted for brevity)

# Show how many emotions are present in the dataset
print(f"Number of Emotions: {num_emotions}")
print("Emotion Labels:")
print(emotion_labels)

# Four example sentences to test emotion detection
sample_texts = [
    "I love this weather! It makes me feel so happy.",
    "Today was a really sad day. I miss my friend.",
    "The terrible service at the restaurant made me furious.",
    "Just another regular day, nothing special.",
]

# Test the model on example sentences
print("\nEmotion Detection Results:")
with torch.no_grad():
    for text in sample_texts:
        inputs = tokenizer(text, return_tensors='pt',
                           padding=True, truncation=True).to(device)
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
        detected_emotions = [emotion_labels[i]
                             for i in range(num_emotions) if probabilities[0][i] > 0.5]
        print(f"Text: {text}")
        print(f"Detected Emotions: {', '.join(detected_emotions)}")
        print()
