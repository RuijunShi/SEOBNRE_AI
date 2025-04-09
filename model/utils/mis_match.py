from pycbc import types
from pycbc.filter import match
import numpy as np
from tqdm import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline as spline


def cal_mismatch(target_amp, target_phase, output_amp, output_phase):
    time_series = np.linspace(0, 10000, 10000) / 4096
    dt = time_series[1] - time_series[0]
    init_mismatch = mismatchclass(dt, time_series)
    len_wavefrom = target_amp.shape[0]

    mismatch_list = []
    for ii in tqdm(range(len_wavefrom)):
        mismatch = init_mismatch.mismatch(
            target_amp[ii], target_phase[ii], output_amp[ii], output_phase[ii]
        )
        mismatch_list.append(mismatch)
    return np.asarray(mismatch_list)


class mismatchclass:
    def __init__(self, dt, time_series):
        self.dt = dt
        self.time_series = time_series

    def mismatch(self, amp1, amp2, pha1, pha2):
        wave_a = amp1 * np.exp(1j * pha1).real
        wave_b = amp2 * np.exp(1j * pha2).real

        a = types.TimeSeries(wave_a, delta_t=self.dt)
        b = types.TimeSeries(wave_b, delta_t=self.dt)
        m, i = match(a, b)
        return m
