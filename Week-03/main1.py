import random


def create_ngrams(text, n):
    ngrams = []
    words = text.split()
    for i in range(len(words) - n + 1):
        ngrams.append(words[i:i + n])
    return ngrams


def build_language_model(text, n):
    ngrams = create_ngrams(text, n)
    model = {}
    for ngram in ngrams:
        prefix = ' '.join(ngram[:-1])
        suffix = ngram[-1]
        if prefix in model:
            model[prefix].append(suffix)
        else:
            model[prefix] = [suffix]
    print(model)
    return model


def generate_text(model, n, max_length=50):
    prefix = random.choice(list(model.keys()))
    print(f"First Prefix:  '{prefix}' ")
    generated_text = prefix.split()
    print(f" First Generated text {generated_text} ")
    while len(generated_text) < max_length:
        print("While loop start")
        if prefix not in model:
            break
        next_word = random.choice(model[prefix])
        print(f"Next word: '{next_word}' ")
        generated_text.append(next_word)
        print(f" Looped Generated text {generated_text} ")
        prefix = ' '.join(generated_text[-n + 1:])
        print(f"final prefix: '{prefix}' ")
    print(' '.join(generated_text))
    return ' '.join(generated_text)


if __name__ == "__main__":
    # Example text for language modeling
    text = "A language model is a type of probabilistic model that assigns probabilities to sequences of words. It is used in various natural language processing tasks."

    # N-gram size
    n = 2

    # Build the language model
    language_model = build_language_model(text, n)

    # Generate new text using the language model
    generated_text = generate_text(language_model, n, max_length=30)

    print("Original Text:")
    print(text)
    print("\nGenerated Text:")
    print(generated_text)
