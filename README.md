# 🔁 RNN and LSTM Implementation from Scratch

This repository contains a step-by-step NumPy-based implementation of **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory Networks (LSTMs)**, along with their forward and backward propagation logic.

---

## 🧠 Architecture Overview

### 🔷 RNN
- One hidden state per time-step
- Simple tanh-based cell
- Forward & backward propagation implemented manually

### 🔶 LSTM
- Four gates: Forget, Input, Cell, Output
- Learns long-term dependencies better
- Full backpropagation through time (BPTT) implementation

---

## ⚙️ Components

### ✅ Forward Pass
- `rnn_cell_forward()` : Computes the hidden state and output prediction for one time-step
- `rnn_forward()` : Iterates over time to get outputs and hidden states for the sequence
- `lstm_cell_forward()` : Computes forward step for one time-step of LSTM
- `lstm_forward()` : Computes the full forward pass for LSTM over sequence

### 🔁 Backward Pass
- `rnn_cell_backward()` : One-step backward pass of RNN
- `rnn_backward()` : Full backward propagation through time for RNN
- `lstm_cell_backward()` : Computes all gate gradients and backprop through one LSTM cell
- `lstm_backward()` : Computes BPTT over the full sequence for LSTM

---

## 📦 File Structure

```bash
├── rnn_utils.py         # Utility functions: softmax, sigmoid, initialization, etc.
├── rnn_lstm_main.py     # Complete RNN & LSTM forward and backward pass
├── data/                # Input examples (optional)
```

---

## 🧪 Testing and Verification

All core functions are unit-tested with NumPy arrays to validate correctness.

```python
assert gradients["dx"].shape == (n_x, m, T_x)
assert gradients["dWf"].shape == (n_a, n_a + n_x)
```

---

## 🔬 Sample Output

```text
a[4][3][6] = 0.2197
c[1][2][1] = -0.2219
gradients["dWax"].shape = (5,3)
gradients["dWc"].shape = (5,8)
```

---

## 📚 References

- Michael Nielsen — [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- Christopher Olah — [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- DeepLearning.AI — Andrew Ng’s [Sequence Models Course](https://www.coursera.org/learn/nlp-sequence-models)
- [TensorFlow RNN Tutorial](https://www.tensorflow.org/tutorials/text/text_classification_rnn)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [CS231n Notes: RNNs](https://cs231n.github.io/recurrent-neural-networks/)
