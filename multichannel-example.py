from tire_cpd.models import multichannel
from tire_cpd.functions import cpd
import data_loader


# ________________  Data Preparation  ________________ #
# Splitting sequential data into windows (windows_TD) and computing the fft (windows_FD) of them
# data is the original time series data, parameters contains the ground truth labels.
# Both are for visualization and evaluation
data, windows_TD, windows_FD, parameters = data_loader.data_parse(30, "timeseries", "change-A3", 40)

# ________________  Model Training  ________________ #
# Training models in both time domain and frequency domain
# Returned results are the extracted time-invariant representations in both branches
# and the reconstruction losses from both branches, which will be used in the postprocessing procedure
shared_AS_TD, shared_B_TD, loss_coherent_TD, loss_incoherent_TD = multichannel.train_model(windows_TD)
shared_AS_FD, shared_B_FD, loss_coherent_FD, loss_incoherent_FD = multichannel.train_model(windows_FD)

# ________________  Change Point Detection  ________________ #
# If information from both domains is used, compute averaged values of reconstruction losses
loss_coherent = (loss_coherent_TD + loss_coherent_FD) / 2
loss_incoherent = (loss_incoherent_TD + loss_incoherent_FD) / 2
# Computing distance between extracted time-invariant representations
dissimilarities = cpd.smoothened_dissimilarity_measures_multichannel(shared_AS_TD, shared_AS_FD, shared_B_TD, shared_B_FD, 40)
# Detecting candidate change points
cpd.show_result_multi_channel(40, dissimilarities, loss_coherent, loss_incoherent)

