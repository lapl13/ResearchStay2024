import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Define the emotion labels
emotion_labels = {0: "happy", 1: "sad", 2: "angry", 3: "neutral"}

# Define a function for text emotion detection


def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    emotion_id = torch.argmax(probabilities, dim=1).item()
    emotion = emotion_labels[emotion_id]
    return emotion


# Test the model with some sample texts
sample_texts = [
    "I'm so happy! Everything is going great.",
    "I feel really sad today. It's been a tough day.",
    "This makes me so angry. I can't believe it!",
    "Just another normal day, nothing special.",
]

for text in sample_texts:
    detected_emotion = detect_emotion(text)
    print(f"Text: {text}")
    print(f"Detected Emotion: {detected_emotion}")
    print()
