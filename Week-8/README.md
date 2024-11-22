# **Week 8: Sequence-to-Sequence (Seq2Seq) Models and Attention Mechanisms**

## **Objectives:**

Week 8 focuses on Sequence-to-Sequence (Seq2Seq) models, a type of neural network architecture designed for tasks that involve mapping input sequences to output sequences. These models are fundamental in tasks such as machine translation and text summarization. The introduction of attention mechanisms further enhances Seq2Seq models by allowing them to focus on relevant parts of the input during prediction.

- Understand the architecture and functionality of Seq2Seq models.
- Explore the role of attention mechanisms in improving model performance.
- Implement Seq2Seq models with attention for tasks such as translation and summarization.

## **Code Examples**

### Example 1 - Seq2Seq with Attention:
This example demonstrates the implementation of a Sequence-to-Sequence (Seq2Seq) model with attention mechanisms. Attention allows the decoder to focus on specific parts of the encoder’s output, improving performance for tasks involving long input sequences.

**Highlights**:
- **Encoder**: Processes the input sequence and outputs a context vector.
- **Attention Mechanism**: Computes weights for each word in the input sequence, indicating its relevance to the current output word.
- **Decoder**: Generates the output sequence, leveraging the context vector and attention weights.

**Applications**:
- Machine translation: Translating a sentence from one language to another.
- Summarization: Creating concise summaries of text while retaining key information.

### Example 2 - Training and Evaluation:
This example provides a detailed look at training Seq2Seq models with attention mechanisms. The training process involves:
- **Data Preparation**: Tokenizing and encoding text sequences for input and output.
- **Loss Function**: Calculating the loss between predicted and actual sequences using cross-entropy loss.
- **Optimization**: Using optimizers such as Adam to minimize loss during training.
- **Evaluation Metrics**: Measuring translation quality using BLEU scores.

**Applications**:
- Text-to-text generation tasks.
- Evaluating translation performance using benchmark datasets.

*Note: For complete implementation details, refer to the repository.*  

## **Professor Provided Readings:**

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

## **Self Researched Readings:**

- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
