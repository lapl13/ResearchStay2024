import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Dummy English to French translation dataset
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# Dummy data
training_pairs = [
    ("I am a student", "Je suis étudiant"),
    ("He likes ice cream", "Il aime la glace"),
    ("She speaks French", "Elle parle français"),
    ("They are happy", "Ils sont heureux"),
    ("We eat dinner", "Nous dînons")
]

# Build vocabulary
input_vocab = set()
output_vocab = set()
for pair in training_pairs:
    input_vocab.update(pair[0].split())
    output_vocab.update(pair[1].split())

input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)

input_word2idx = {word: idx for idx, word in enumerate(input_vocab)}
output_word2idx = {word: idx for idx, word in enumerate(output_vocab)}

# Convert sentences to tensors
def sentence_to_tensor(sentence, vocab):
    return torch.tensor([vocab[word] for word in sentence.split()])

training_data = [(sentence_to_tensor(pair[0], input_word2idx), sentence_to_tensor(pair[1], output_word2idx))
                 for pair in training_pairs]

# Encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return output, hidden

# Attention layer
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, encoder_outputs, decoder_hidden):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        decoder_hidden = decoder_hidden.unsqueeze(2)
        attn_weights = torch.bmm(encoder_outputs, decoder_hidden).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_outputs)
        return context

# Decoder model with attention
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(0)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        context = self.attention(encoder_outputs, hidden)
        output = torch.cat((output, context), dim=2)
        output = F.log_softmax(self.out(output.squeeze(0)), dim=1)
        return output, hidden

# Encoder-Decoder model with attention
class EncoderDecoderAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoderAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        target_len = trg.shape[0]
        target_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(trg.device)
        encoder_outputs, hidden = self.encoder(src)

        decoder_input = trg[0, :]
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[t] = output
            use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False
            decoder_input = trg[t] if use_teacher_forcing else output.argmax(1)
        return outputs

# Training the model
encoder_hidden_size = 64
decoder_hidden_size = 64

encoder = Encoder(input_vocab_size, encoder_hidden_size)
decoder = Decoder(decoder_hidden_size, output_vocab_size)

model = EncoderDecoderAttention(encoder, decoder)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    for src, trg in training_data:
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Translating new sentences
def translate_sentence(sentence, model, input_vocab, output_vocab):
    with torch.no_grad():
        model.eval()
        src_tensor = sentence_to_tensor(sentence, input_vocab)
        src_tensor = src_tensor.unsqueeze(1).to(next(model.parameters()).device)
        encoder_outputs, hidden = model.encoder(src_tensor)
        trg_indexes = [output_word2idx['<SOS>']]
        for _ in range(10):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(next(model.parameters()).device)
            with torch.no_grad():
                output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == output_word2idx['<EOS>']:
                break
        trg_tokens = [output_vocab[idx] for idx in trg_indexes]
        return trg_tokens[1:]

# Test translation
test_sentence = "We are happy"
translated_sentence = translate_sentence(test_sentence, model, input_word2idx, output_word2idx)
print(f"English: {test_sentence}")
print(f"French: {' '.join(translated_sentence)}")
