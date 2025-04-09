import torch
import torch.nn as nn
from .plot_test import plot_train


def test(model, test_ds, device, save_image):
    test_loss = 0
    for ii, batch in enumerate(test_ds):
        par, target_waveforms_amp, target_waveforms_pha, mask = batch
        par = par.to(device).float()
        mask = mask.to(device).float()
        target_waveforms_amp = target_waveforms_amp.to(device).float()
        generated_waveforms_amp = model(par, target_waveforms_amp)
        test_loss += model.total_loss.item()

    test_loss = test_loss / (ii + 1)

    plot_train(generated_waveforms_amp, target_waveforms_amp, save_image)

    return test_loss
