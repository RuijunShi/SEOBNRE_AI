import os, h5py
import numpy as np

# import torchvision.transforms.functional as F
# from torch.utils.data import DataLoader
import torch
from scipy.interpolate import CubicSpline
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5_file,
        dataset_name,
        waveform_len,
        mode,
        interpolation=False,
        device=None,
        array_len=20000
    ):
        super(Dataset, self).__init__()

        self.hdf5_file = hdf5_file
        self.dataset_name = dataset_name
        self.load_mode(mode)
        self.array_len=array_len

        if interpolation:
            with h5py.File(self.hdf5_file, "r") as f:
                par, amp, pha = self.loading_dataset(f, self.dataset_name)

            par = np.asarray(list(par))

            par_empty = np.zeros((par.shape[0], par.shape[1] - 1))
            par_empty[:, 0] = self.map_to_minus1_1(par[:, 0] / par[:, 1], 1, 5)
            par_empty[:, 1] = self.map_to_minus1_1(par[:, 2], 0, 0.2)
            par_empty[:, 2] = self.map_to_minus1_1(par[:, 3], -0.6, 0.6)
            par_empty[:, 3] = self.map_to_minus1_1(par[:, 4], -0.6, 0.6)
            par = par_empty
            self.length = par.shape[0]

            len_of_all_waveform = np.zeros(self.length)
            new_wavefrom_amp = []
            new_wavefrom_pha = []

            for ii in tqdm(range(par.shape[0])):
                tmp_amp = amp[ii]
                tmp_pha = pha[ii]

                merge_index = np.argmax(tmp_amp)
                all_len = merge_index + int(merge_index * 0.1)
                if self.mode == 2 or self.mode == 3:
                    len_of_all_waveform[ii] = all_len
                else:
                    # ---- interpolation ---
                    x = np.linspace(0, all_len, all_len)

                    tmp_amp, tmp_pha = self.waveform_cutv2(tmp_pha, tmp_amp, all_len)

                    amp[ii] = tmp_amp
                    pha[ii] = tmp_pha
                    y_new_amp = self.interpolation(x, tmp_amp, all_len, waveform_len)
                    y_new_pha = self.interpolation(x, tmp_pha, all_len, waveform_len)

                    # ---- append to dataset ---
                    new_wavefrom_amp.append(y_new_amp[:])
                    new_wavefrom_pha.append(y_new_pha[:])

                    len_of_all_waveform[ii] = all_len

        else:
            with h5py.File(self.hdf5_file, "r") as f:
                par = f[self.dataset_name]["par"][:]
                new_wavefrom_amp = f[self.dataset_name]["label"][:]

                self.length = par.shape[0]
                amp = new_wavefrom_amp
                pha = new_wavefrom_amp
                new_wavefrom_pha = new_wavefrom_amp

        self.par = par
        if self.mode == 0:
            self.label = np.asarray(new_wavefrom_amp)
            if device is not None:
                self.par = torch.asarray(par).to(device)
                self.label = torch.asarray(self.label).to(device)
            del new_wavefrom_pha, amp, pha

        elif self.mode == 1:
            self.label = np.asarray(new_wavefrom_pha)
            if device is not None:
                self.par = torch.asarray(par).to(device)
                self.label = torch.asarray(np.asarray(self.label)).to(device)
            del new_wavefrom_amp, amp, pha

        elif self.mode == 2:
            self.label = len_of_all_waveform
            if device is not None:
                self.par = torch.asarray(par).to(device)
                self.label = torch.asarray(np.asarray(self.label)).to(device)
            del new_wavefrom_amp, new_wavefrom_pha

        elif self.mode == 3:
            self.par = par
            self.len_of_all_waveform = len_of_all_waveform
            self.amp_raw = amp
            self.pha_raw = pha

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        parameter = self.par[idx]
        if self.mode == 0 or self.mode == 1 or self.mode == 2:
            label = self.label[idx]
            return parameter, label
        elif self.mode == 3:
            tem_amp = np.zeros(self.array_len)
            tmp_pha = np.zeros(self.array_len)
            amp1 = self.amp_raw[idx]
            pha1 = self.pha_raw[idx]
            tem_amp[: amp1.shape[0]] = amp1
            tmp_pha[: amp1.shape[0]] = pha1

            amp_raw = tem_amp  # self.amp_raw[idx]
            pha_raw = tmp_pha  # self.pha_raw[idx]
            len_of_signal_wave = self.len_of_all_waveform[idx]
            return parameter, amp_raw, pha_raw, len_of_signal_wave

    def interpolation(self, x, y, all_len, sample_len):
        cs = CubicSpline(x, y)
        target_len = np.linspace(0, all_len, sample_len)
        y_new = cs(target_len)
        return y_new

    def waveform_cut(self, tmp_pha, tmp_amp, all_len):
        index_end = np.argwhere(tmp_amp < 1e-6)[0] - 1

        tmp_zero_amp = np.zeros(all_len)
        tmp_ones_pha = np.ones(all_len + 1) * tmp_pha[index_end]

        tmp_zero_amp[: tmp_amp.shape[0]] = tmp_amp
        tmp_ones_pha[:index_end] = tmp_pha[:index_end]

        tmp_amp = tmp_zero_amp
        tmp_pha = tmp_ones_pha
        return tmp_amp, np.diff(tmp_pha)

    def waveform_cutv2(self, tmp_pha, tmp_amp, all_len):
        index_end = int(np.argwhere(tmp_amp < 1e-6)[0] - 1)

        tmp_zero_amp = np.zeros(all_len)
        tmp_ones_pha = np.ones(all_len) * tmp_pha[index_end]

        if all_len > tmp_amp.shape[0]:
            tmp_zero_amp[: tmp_amp.shape[0]] = tmp_amp
            tmp_ones_pha[:index_end] = tmp_pha[:index_end]

            tmp_amp = tmp_zero_amp
            tmp_pha = tmp_ones_pha
        else:
            tmp_amp = tmp_amp[:all_len]
            tmp_ones_pha[:index_end] = tmp_pha[:index_end]
            tmp_pha = tmp_ones_pha
        return tmp_amp, tmp_pha

    def loading_dataset(self, f, group):
        print(">>>>>>> loading dataset >>>>>>>")
        k = 0
        for ii in tqdm(f[group]["amp"].keys()):
            if k == 0:
                par = f[group]["par"][ii][:]
                amp = f[group]["amp"][ii][:]
                pha = f[group]["phase"][ii][:]
            else:
                par = np.concatenate([par, f[group]["par"][ii][:]], axis=0)
                amp = np.concatenate([amp, f[group]["amp"][ii][:]], axis=0)
                pha = np.concatenate([pha, f[group]["phase"][ii][:]], axis=0)
            k += 1
        return par, amp, pha

    def load_mode(self, mode):
        if mode == "train_amp":
            self.mode = 0
        elif mode == "train_pha":
            self.mode = 1
        elif mode == "train_wave_len":
            self.mode = 2
        elif mode == "test":
            self.mode = 3
        else:
            raise Exception(
                "The input must be 'train_amp', 'train_pha', 'train_wave_len', or 'test'!!!"
            )

    def write_hdf5_interpolation(self, hdf5_out):
        par = self.par
        amp = self.label
        if self.dataset_name == "train":
            h5_w = "w"
        else:
            h5_w = "a"
        with h5py.File(hdf5_out, h5_w) as f:
            ds = f.create_group(self.dataset_name)
            ds.create_dataset("par", par)
            ds.create_dataset("amp", amp)

    def map_to_minus1_1(self, x, old_min=0, old_max=5):
        return -1 + (2) * (x - old_min) / (old_max - old_min)
