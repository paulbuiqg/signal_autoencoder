# Signal Autoencoder

A neural network autoencoder for multichannel, one-dimensional signals of variable length.

The encoder pipes three 1D-convolution layers followed by a ReLU, and two recurrent layers (LSTM); the decoder goes the other way.

Schematically, the encoder design is:
```
    input
->  Conv1d(n_channel_in, n_conv_channel_1)
->  ReLU
->  Conv1d(n_conv_channel_1, n_conv_channel_2)
->  ReLU
->  Conv1d(n_conv_channel_2, n_conv_channel_3)
->  ReLU
->  LSTM(n_conv_channel_3, lstm_hidden_size)
->  output, hidden_state, cell_state
```
and the decoder design is
```
    output, hidden_state, cell_state
->  LSTM(lstm_hidden_size, lstm_hidden_size)
->  Linear(lstm_hidden_size, n_conv_channel_3)
->  ReLU
->  Conv1d(n_conv_channel_3, n_conv_channel_2)
->  ReLU
->  Conv1d(n_conv_channel_2, n_conv_channel_1)
->  ReLU
->  Conv1d(n_conv_channel_1, n_channel_in)
```

The code produced by the encoder is the (flattened) hidden state outputed from its LSTM layers.

To do next:

- find a corresponding dataset
- training
