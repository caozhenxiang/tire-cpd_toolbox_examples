from tire_cpd.models import diamond
from tire_cpd.functions import cpd
import numpy as np
import data_loader


# ________________  Data Preparation  ________________ #
# Splitting sequential data into windows (windows_TD) and computing the fft (windows_FD) of them
# data is the original time series data, parameters contains the ground truth labels.
# Both are for visualization and evaluation
data, windows_TD, windows_FD, parameters = data_loader.data_parse(nfft=30,
                                                                  norm_mode="timeseries",
                                                                  dataset="mean0",
                                                                  window_size=40)
# Before feeding windows to the model, ensure they are 3-dimensional
# (nr_windows_per_channel * window_size * nr_channels)
if len(data.shape) == 1:
        time_series = np.expand_dims(data, axis=-1)
        windows_TD = np.expand_dims(windows_TD, axis=-1)
        windows_FD = np.expand_dims(windows_FD, axis=-1)

# ________________  Model Training  ________________ #
# Training models in both time domain and frequency domain
# Returned results are the extracted time-invariant representations
shared_features_TD = diamond.train_model(windows_TD, window_size=40)
shared_features_FD = diamond.train_model(windows_FD, window_size=40)

# ________________  Change Point Detection  ________________ #
# Computing distance between extracted time-invariant representations
dissimilarities = cpd.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD,
                                                               domain="both", window_size=40)
# Detecting candidate change points
cpd.show_result(dissimilarities, window_size=40)

