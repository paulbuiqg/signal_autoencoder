# Signal Autoencoder (WIP)

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

## Study case: 3-channel seismic signals

I apply this autoencoder to a dataset of signals of seismic waveforms measured by 3-channel seismometers. The three channels correspond to the ground motion along the North-South, East-West and vertical axes.

The data is obtained from the [Northern California Earthquake Data Center](https://ncedc.org/). Their web service is [queried](https://service.ncedc.org/fdsnws/event/1/query?minmag=5&maxmag=9) for seismic events with a magnitude between 5 and 9.

...

## How to use

- Go to the repo root directory
- Install required libraries: `pip install -r requirements.txt`
- ...
- For unit testing, run: `pytest`
  
...
