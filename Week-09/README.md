# **Week 9: Attention Mechanisms in Sequence-to-Sequence Models**

## **Objectives:**

Week 9 explores attention mechanisms in Sequence-to-Sequence (Seq2Seq) models, a key enhancement that addresses the limitations of traditional Seq2Seq architectures. Attention allows the model to focus on relevant parts of the input sequence during decoding, making it particularly effective for tasks involving long or complex sequences.

- Understand the functionality and types of attention mechanisms (e.g., Bahdanau and Luong attention).
- Implement attention mechanisms in Seq2Seq models for tasks like machine translation or summarization.
- Analyze the impact of attention mechanisms on model performance and sequence quality.

## **Code Examples**

### Example 1 - Bahdanau Attention:
This example demonstrates the implementation of Bahdanau Attention in a Seq2Seq model. Bahdanau Attention, also known as additive attention, dynamically computes weights for input sequence elements, allowing the decoder to focus on relevant parts during each time step.

**Highlights**:
- Attention weights are calculated using a feedforward neural network.
- The weights indicate the importance of each input word to the current output word.
- Improves the model's ability to handle long input sequences.

**Applications**:
- Machine translation: Generating accurate translations for complex sentences.
- Text summarization: Producing concise summaries by focusing on key points in the text.

### Example 2 - Luong Attention:
This example focuses on implementing Luong Attention, a simpler and more computationally efficient alternative to Bahdanau Attention. It computes alignment scores directly using dot-product or general methods.

**Highlights**:
- Alignment scores are calculated as a similarity measure between encoder outputs and the decoder state.
- Suitable for tasks requiring faster computations.

**Applications**:
- Real-time language translation.
- Sentence simplification tasks.

*Note: For full implementation details, refer to the repository.*  

## **Professor Provided Readings:**

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

## **Self Researched Readings:**

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Luong Attention Mechanism](https://arxiv.org/abs/1508.04025)
