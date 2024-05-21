# Signal Autoencoder

A neural network autoencoder for multichannel, one-dimensional signals of variable length.

The encoder pipes one convolutional block with three 1D-convolutions layers, and one recurrent block with a two-layered LSTM module; the decoder goes the other way.

Schematically, the encoder architecture is:
```
    input
->  Conv1d(n_channel_in, n_conv_channel_1)
->  ReLU
->  Conv1d(n_conv_channel_1, n_conv_channel_2)
->  ReLU
->  Conv1d(n_conv_channel_2, n_conv_channel_3)
->  LSTM(n_conv_channel_3, lstm_hidden_size)
->  output, hidden_state, cell_state
```
and the decoder architecture is
```
    output, hidden_state, cell_state
->  LSTM(lstm_hidden_size, lstm_hidden_size)
->  Linear(lstm_hidden_size, n_conv_channel_3)
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
