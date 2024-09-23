# **Week 6: Long short-term memory (LSTM) and gated recurrent units (GRU)**


## **Objectives:**

Week 6 focuses on Long Short-Term Memory (LSTM) networks, and Gated Recurrent Units (GRU). LSTM networks are a type of recurrent neural network (RNN) designed to overcome the log term dependency problem that tradtional RNN's have, GRU's are also a type of RNN, similar to LSTM, but with a simpler and more streamlined architecture and faster training speeds.
 
## Code Examples

### Example 1 - LSTM: 
In this example, we explore the Long Short-Term Memory (LSTM) neural network, which is a variant of recurrent neural networks (RNNs). LSTMs are particularly useful for sequential data tasks, such as time series forecasting or natural language processing, where there are long-term dependencies between data points.

Data Generation: We begin by generating synthetic sequential data for our LSTM model. The data consists of two sequences, each containing ten consecutive numbers. The target is the third number in each sequence. We convert this data into PyTorch tensors, which is the format required for processing with PyTorch.

LSTM Model Architecture: The LSTM model is defined using the LSTMModel class. Inside the class, we set up the LSTM layer, which takes a sequence of one-dimensional inputs and processes them in a time-ordered manner. We also include a fully connected layer to predict the next number in the sequence. The model architecture captures the patterns and dependencies present in the input sequences.

Forward Pass: The forward method within the LSTMModel class describes how data flows through the model. During the forward pass, the LSTM processes the input sequences, capturing important information over time. The hidden state and cell state are initialized at the beginning of each forward pass and updated as the LSTM processes the data.

Loss Function and Optimizer: For training, we define the Mean Squared Error (MSE) loss function to quantify the difference between predicted and actual values. To optimize the model's parameters during training, we use the Adam optimizer, which adapts the learning rate based on the gradients of the model's parameters.

Training Loop: We run a loop for a specified number of epochs to train the LSTM model. In each epoch, we forward propagate the input sequences through the model, compute the loss, perform backpropagation to calculate gradients, and update the model's parameters using the optimizer. The goal is to minimize the loss and improve the model's prediction accuracy.

Testing the Trained Model: After training, we evaluate the LSTM model's performance on a test sequence, [10, 11]. The trained LSTM predicts the next number in the sequence based on the learned patterns from the training data.

### Example 2 - GRU: 
In this example, we explore the Gated Recurrent Unit (GRU), which is a variant of recurrent neural networks (RNNs). Like LSTMs, GRUs are particularly useful for sequential data tasks, but they have fewer parameters and are often computationally faster.

Data Generation: We start by generating synthetic sequential data, which consists of two sequences, each containing ten consecutive numbers. The target is the third number in each sequence. We convert this data into PyTorch tensors, preparing it for processing with PyTorch.

GRU Model Architecture: The GRU model is defined using the GRUModel class. Inside this class, we set up the GRU layer, which takes a sequence of one-dimensional inputs and processes them in a time-ordered manner. Additionally, we include a fully connected layer to predict the next number in the sequence. The model architecture captures the patterns and dependencies present in the input sequences.

Forward Pass: The forward method within the GRUModel class describes how data flows through the model. During the forward pass, the GRU processes the input sequences, capturing important information over time. The hidden state is initialized at the beginning of each forward pass and updated as the GRU processes the data.

Loss Function and Optimizer: For training, we define the Mean Squared Error (MSE) loss function to measure the difference between predicted and actual values. To optimize the model's parameters during training, we use the Adam optimizer, which adapts the learning rate based on the gradients of the model's parameters.

Training Loop: We run a loop for a specified number of epochs to train the GRU model. In each epoch, we forward propagate the input sequences through the model, compute the loss, perform backpropagation to calculate gradients, and update the model's parameters using the optimizer. The goal is to minimize the loss and improve the model's prediction accuracy.

Testing the Trained Model: After training, we evaluate the GRU model's performance on a test sequence, [10, 11]. The trained GRU predicts the next number in the sequence based on the learned patterns from the training data.


## **Professor Provided Readings:**

[Long Sort-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

[Tutorial on GRU](https://d2l.ai/chapter_recurrent-modern/gru.html)

[Using LSTM and GRU neural network methods for traffic flow prediction](https://www.researchgate.net/profile/Li-Li-86/publication/312402649_Using_LSTM_and_GRU_neural_network_methods_for_traffic_flow_prediction/links/5c20d38d299bf12be3971696/Using-LSTM-and-GRU-neural-network-methods-for-traffic-flow-prediction.pdf)


## **Self Researched Readings:**

[LSTM and GRU Neural Networks as Models of Dynamical Processes Used in Predictive Control: A Comparison of Models Developed for Two Chemical Reactors]([https://www.mdpi.com/2076-3417/13/5/3186](https://www.mdpi.com/1424-8220/21/16/5625))
