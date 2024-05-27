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

The embedding produced by the encoder is the (flattened) hidden state outputed from its LSTM layers.

## Study case: 3-channel seismic signals

I apply this autoencoder to a dataset of signals of seismic waveforms measured by 3-channel seismometers. The three channels correspond to the ground motion along the North-South, East-West and vertical axes.

The data is obtained from the [Northern California Earthquake Data Center](https://ncedc.org/) (NCEDC). Their web service is [queried](https://service.ncedc.org/fdsnws/event/1/query?minmag=5&maxmag=9) for seismic events with a magnitude between 5 and 9.

<p align="center">
  <img src="https://github.com/paulbuiqg/signal_autoencoder/blob/main/viz/HNE.png" />
</p>
<p align="center">
  <img src="https://github.com/paulbuiqg/signal_autoencoder/blob/main/viz/HNN.png" />
</p>
<p align="center">
  <img src="https://github.com/paulbuiqg/signal_autoencoder/blob/main/viz/HNZ.png" />
</p>


## How to use

- Go to the repo root directory
- Download and install packages: `pip install -r requirements.txt`
- Install local package: `pip install -e .`
- For unit testing, run: `pytest`
- To download the seismic data from NCEDC, run: `python3 scripts/seismic_downloading.py`
- To train the autoencoder and compute the seismic signal embeddings, run: `python3 scripts/seismic_autoencoding.py`

Notebooks:
- Generic usage of the signal autoencoder: `notebooks/demo.ipynb`
- (Naive) seismic event signal analysis: `notebooks/seismic_analysis.ipynb`