import sys
sys.path.append("..")
import torch
import numpy as np
from utils.model_class import Interpolation as Interpolation, Par_MergeTime
from scipy.interpolate import CubicSpline

class GenerationWaveform:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        ## loading checkpoint ...
        self.model_par_to_amp = Interpolation(4, 1024, 8, 1024).to(self.device)
        self.model_par_to_amp.load_state_dict(
            torch.load(
                "./check_point/sample_1024_4096Hz_amp_debug_v6_par_nor.pth",
                map_location=self.device,
            )["model"],
            strict=True,
        )

        self.model_par_to_pha = Interpolation(4, 1024, 8, 1024).to(self.device)
        self.model_par_to_pha.load_state_dict(
            torch.load(
                "./check_point/sample_1024_4096Hz_pha_debug_v6_par_nor_cut_rd.pth",
                map_location=self.device,
            )["model"],
            strict=True,
        )

        self.model_par_to_merge_time = Par_MergeTime(4, 1024, v2=False).to(self.device)
        self.model_par_to_merge_time.load_state_dict(
            torch.load(
                "./check_point/sample_1024_4096Hz_merge_time_4par_4layer_L1loss.pth",
                map_location=self.device,
            )["model"],
            strict=True,
        )

        self.model_par_to_amp.eval()
        self.model_par_to_pha.eval()
        self.model_par_to_merge_time.eval()
        print("Model loaded successfully!")

    def generation(self, par):
        if len(par.shape) == 1:
            par = np.expand_dims(par, axis=0)
        par = torch.as_tensor(self.normal_par(par)).to(self.device).float()
        
        with torch.no_grad():
            y_gen_amp = self.model_par_to_amp(par)
            y_gen_pha = self.model_par_to_pha(par)
            len_wave  = self.model_par_to_merge_time(par)
            len_wave = np.asarray(len_wave.cpu().numpy()/2, dtype=int)

        y_new_amp = []
        y_new_pha = []
        for jj in range(y_gen_amp.shape[0]):
            len_w = len_wave[jj]
            x = np.linspace(0, len_w, 1024)
            target_len = np.linspace(0, int(len_w), int(len_w))

            y_amp = y_gen_amp[jj].cpu().numpy()
            cs_amp = CubicSpline(x, y_amp)
            y_new_amp.append(cs_amp(target_len))

            y_pha = y_gen_pha[jj].cpu().numpy()
            cs_amp = CubicSpline(x, y_pha)
            y_new_pha.append(cs_amp(target_len))

        return y_new_amp, y_new_pha

    def normal_par(self, par):
        if par.shape[1] == 5:
            par_empty = np.zeros((par.shape[0], par.shape[1] - 1))
            par_empty[:, 0] = self.map_par_to1_1(par[:, 0] / par[:, 1], 1, 5)
            par_empty[:, 1] = self.map_par_to1_1(par[:, 2], 0, 0.2)
            par_empty[:, 2] = self.map_par_to1_1(par[:, 3], -0.6, 0.6)
            par_empty[:, 3] = self.map_par_to1_1(par[:, 4], -0.6, 0.6)
            par = par_empty
        elif par.shape[1] == 4:
            par_empty = np.zeros((par.shape[0], par.shape[1]))
            par_empty[:, 0] = self.map_par_to1_1(par[:, 0], 1, 5)
            par_empty[:, 1] = self.map_par_to1_1(par[:, 1], 0, 0.2)
            par_empty[:, 2] = self.map_par_to1_1(par[:, 2], -0.6, 0.6)
            par_empty[:, 3] = self.map_par_to1_1(par[:, 3], -0.6, 0.6)
        return par_empty
    
    def map_par_to1_1(self, x, old_min=0, old_max=5):
        return -1 + (2) * (x - old_min) / (old_max - old_min)

    def map_to_minus1_1(self, x, new_min=0, new_max=0.2):
        return new_min + (new_max - new_min) * (x + 1) / 2