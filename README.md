Examples for TIRE-cpd toolbox 
===============================

Toolbox for time-invariant representation autoencoder approach (TIRE) for change point detection (CPD) task. Including three models: TIRE model with diamond loss [1], multi-channel TIRE model [2], and multi-view TIRE model [3].

The authors of these papers are:

- [Zhenxiang Cao](https://www.esat.kuleuven.be/stadius/person.php?id=2380) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Nick Seeuws](https://www.esat.kuleuven.be/stadius/person.php?id=2318) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Maarten De Vos](https://www.esat.kuleuven.be/stadius/person.php?id=203) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven and Dept. Development and Regeneration, KU Leuven)
- [Alexander Bertrand](https://www.esat.kuleuven.be/stadius/person.php?id=331) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)

All authors are affiliated to [LEUVEN.AI - KU Leuven Institute for AI](https://ai.kuleuven.be). 

# 1. Installation
To install the tire-cpd package, one can easily run the following code to obtain it from Pypi.org:
```
pip3 install tire-cpd
```

# 2. Usage
Here we simply explain the models contained in tire-cpd toolbox and their main functions:

## 2.1 Multichannel
Implementation of MC-TIRE model [2]. This model is designed for multi-channel time series deliberately and can considerate the inter-channel coherence (spatial) structure explicitly. 

Use``` from tire_cpd.models import multichannel ``` to import this model.

Use``` from tire_cpd.functions import cpd ``` to import the change point detection functions.

### 2.1.1 model training
Use``` multichannel.train_model(windows, loss_weight_share_AS, loss_weight_share_B, loss_weight_uncor, n_filter, verbose, enable_summary, rank, nr_epochs, nr_patience) ``` to train the MC-TIRE model.

- windows: 3-dimensional tensors with size of (nr_windows_per_channel, window_size, nr_channels). Input time windows from either time domain or frequency domain.
- loss_weight_share_AS: scalar, defaulted value=1e-2. Determine the weight of time-invariant regularization loss in the coherence branch. Refer to [2] for more details.
- loss_weight_share_B: scalar, defaulted value=1e-2. Determine the weight of time-invariant regularization loss in the residual branch. Refer to [2] for more details.
- loss_weight_uncor: scalar, defaulted value=1e-1. Determine the weight of decorrelation loss. Refer to [2] for more details.
- n_filter: scalar, defaulted value=2. Determine the number of time-invariant features in the TIRE models in the residual branch. Refer to [2] for more details.
- verbose: scalar, can be set as 0, 1, or 2. Verbosity mode. 0 = silent, 1 = one line per epoch, 2 = more training details.
- enable_summary: boolean. Enable the summary of used model.
- rank: scalar, defaulted value=1. Determine the rank of low rank approximation used in the coherence branch. Refer to [2] for more details.
- nr_epochs: scalar, defaulted value=200. Determine the number of training epochs.
- nr_patience: scalar, defaulted value=10. Determine the patience for early stopping in the training procedure.

Returned variables are:

- encoded_windows_AS: 2-dimensional matrix with size of (nr_windows_per_channel, nr_features in coherence branch). Extracted time-invariant features from the coherence branch. Refer to [2] for more details.
- encoded_windows_B: 2-dimensional matrix with size of (nr_windows_per_channel, nr_features in residual branch). Extracted time-invariant features from the residual branch. Refer to [2] for more details.
- rec_AS_E_var: 1-dimensional vector with sice of nr_channels. The variance of reconstruction error energy in the coherence branch. 
- rec_B_E_var: 1-dimensional vector with sice of nr_channels. The variance of reconstruction error energy in the residual branch. 

### 2.1.2 change point detection
Use``` dissimilarities = cpd.smoothened_dissimilarity_measures_multichannel(encoded_windows_AS_TD, encoded_windows_AS_FD, encoded_windows_B_TD, encoded_windows_B_FD, window_size) ``` to computing distance between extracted time-invariant representations. If only information in one specific domain is used (either time domain (TD) or frequency domain (FD)), set the variables from the other domain as None.

Use``` cpd.show_result_multi_channel(window_size, dissimilarities, rec_AS_E_var, rec_B_E_var) ``` to identify change points. If models are trained in both domains, use the averaged values of rec_AS_E_var, rec_B_E_var from both domains.

## 2.2 Diamond
Implementation of diamond loss TIRE model [1]. This model is designed for change point detection in low-dimensional time series. If the domain-specific knowledge is unknown, the multiview model is more recommended. 

Use``` from tire_cpd.models import diamond ``` to import this model.

Use``` from tire_cpd.functions import cpd ``` to import the change point detection functions.

### 2.2.1 model training
Use``` diamond.train_model(windows, window_size, enable_summary, verbose, nr_epochs, nr_patience) ``` to train the diamond loss TIRE model.
- windows: 3-dimensional tensors with size of (nr_windows_per_channel, window_size, nr_channels). Input time windows from either time domain or frequency domain.
- window_size: scalar. Determine the selected window size.
- verbose: scalar, can be set as 0, 1, or 2. Verbosity mode. 0 = silent, 1 = one line per epoch, 2 = more training details.
- enable_summary: boolean. Enable the summary of used model.
- nr_epochs: scalar, defaulted value=200. Determine the number of training epochs.
- nr_patience: scalar, defaulted value=10. Determine the patience for early stopping in the training procedure.

As a result, the extracted time-invariant features are returned:

- encoded_windows: 2-dimensional matrix with size of (nr_windows_per_channel, nr_features). Refer to [2] for more details.

### 2.2.2 change point detection
Use``` dissimilarities = smoothened_dissimilarity_measures(encoded_windows_TD, encoded_windows_FD, domain, window_size) ``` to computing distance between extracted time-invariant representations. If only information in one specific domain is used (either time domain (TD) or frequency domain (FD)), set the variable domain as either TD or FD.

Use``` cpd.show_result(dissimilarities, window_size) ``` to identify change points. 

## 2.3 Multiview
Implementation of multi-view TIRE model [3]. This model is designed for change point detection in low-dimensional time series, when the domain-specific knowledge is unknown.

Use``` from tire_cpd.models import multiview ``` to import this model.

Use``` from tire_cpd.functions import cpd ``` to import the change point detection functions.

### 2.3.1 model training
Use``` multiview.train_model(windows, window_size, enable_summary, verbose, nr_epochs, nr_patience) ``` to train the multi-view TIRE model.
- windows: 3-dimensional tensors with size of (nr_windows_per_channel, window_size_TD+window_size_FD, nr_channels). Input time windows from both domains.
- window_size: scalar. Window size in the time domain (TD).
- verbose: scalar, can be set as 0, 1, or 2. Verbosity mode. 0 = silent, 1 = one line per epoch, 2 = more training details.
- enable_summary: boolean. Enable the summary of used model.
- nr_epochs: scalar, defaulted value=200. Determine the number of training epochs.
- nr_patience: scalar, defaulted value=20. Determine the patience for early stopping in the training procedure.

As a result, the extracted time-invariant features are returned:

- shared_features_both: 2-dimensional matrix with size of (nr_windows_per_channel, nr_features). Refer to [3] for more details.

### 2.3.2 change point detection
Use``` dissimilarities = cpd.smoothened_dissimilarity_measures_multiview(shared_features_both, window_size_TD) ``` to computing distance between extracted time-invariant representations.

Use``` cpd.show_result(dissimilarities, window_size) ``` to identify change points. 


# References
- [1] Cao, Z., Seeuws, N., De Vos, M. and Bertrand, A., 2023. A novel loss for change point detection models with time-invariant representations. IEEE Signal Processing Letters, 30, pp.1737-1741.
- [2] Cao, Z., Seeuws, N., De Vos, M. and Bertrand, A., 2023. Change Point Detection in Multi-Channel Time Series Via a Time-Invariant Representation. IEEE Transactions on Knowledge and Data Engineering.
- [3] Cao, Z., Seeuws, N., De Vos, M. and Bertrand, A., 2024. A Multi-view Extension for Change Point Detection via Time-invariant Representations. Proceedings of EUSIPCO 2024.