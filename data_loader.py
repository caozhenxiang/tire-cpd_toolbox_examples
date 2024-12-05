from tire_cpd.functions import preprocessing, utils
import numpy as np
import os

def data_parse(nfft, norm_mode, dataset, window_size):
    path = os.path.abspath(os.getcwd())

    if ("mean" in dataset) or ("JM" in dataset):
        data_name = dataset + "_data.txt"
        label_name = dataset + "_labels.txt"
        data = np.genfromtxt(path + "/Data/used_data/" + dataset[:-1] + "/" + data_name, delimiter=" ")
        parameters = np.genfromtxt(path + "/Data/used_data/" + dataset[:-1] + "/" + label_name, delimiter=" ")
        windows_td = preprocessing.ts_to_windows(data, 0, window_size, 1, normalization="timeseries")
        windows_FD = utils.calc_fft(windows_td, nfft, norm_mode=norm_mode)
        windows_TD = utils.norm_windows(windows_td)
        return data, windows_TD, windows_FD, parameters

    elif "change" in dataset:
        folder = dataset[:-1]
        data = np.genfromtxt(path + "/Data/used_data/" + folder + "/" + dataset + "_data.txt", delimiter=" ")
        parameters = np.genfromtxt(path + "/Data/used_data/" + folder + "/" + dataset + "_labels.txt", delimiter=" ")
        for idx in range(np.shape(data)[1]):
            windows_td = preprocessing.ts_to_windows(data[:, idx], 0, window_size, 1, normalization="timeseries")
            windows_fd = utils.calc_fft(windows_td, nfft, norm_mode=norm_mode)
            windows_td = utils.norm_windows(windows_td)
            if idx == 0:
                windows_TD_bu = [windows_td]
                windows_FD_bu = [windows_fd]
            else:
                windows_TD_bu.append(windows_td)
                windows_FD_bu.append(windows_fd)
        windows_TD = np.transpose(np.array(windows_TD_bu), [1, 2, 0])
        windows_FD = np.transpose(np.array(windows_FD_bu), [1, 2, 0])
        return data, windows_TD, windows_FD, parameters

    else:  # TODO: extend to further dataset
        return 0, 0, 0, 0