import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_train(output, waveform, name):
    index = 1
    plt.figure(figsize=(8, 5), dpi=300)
    plt.subplot(211)
    plt.plot(waveform[index].cpu().detach().numpy(), label="target")
    plt.plot(output[index].cpu().detach().numpy(), label="generation")
    plt.ylabel("waveform amplitude")
    plt.legend()

    plt.subplot(212)
    plt.plot(
        np.abs(
            output[index].cpu().detach().numpy()
            - waveform[index].cpu().detach().numpy()
        )
    )
    plt.ylabel("residual amplitude")
    plt.xlabel("time[s*dt]")
    plt.savefig(name)
    plt.close()


def plot_train_hp(output_pha, target_pha, target_amp, name):
    index = 1
    plt.figure(figsize=(8, 5), dpi=300)
    plt.subplot(211)
    hp_output = (target_amp[index] * torch.exp(1j * output_pha[index])).real
    hp_target = (target_amp[index] * torch.exp(1j * target_pha[index])).real
    plt.plot(hp_output.cpu().detach().numpy())
    plt.plot(hp_target.cpu().detach().numpy())
    plt.ylabel("waveform amplitude")

    plt.subplot(212)
    plt.plot(
        np.abs(hp_output.cpu().detach().numpy() - hp_target.cpu().detach().numpy())
    )
    plt.ylabel("residual amplitude")
    plt.xlabel("time[s*dt]")
    plt.savefig(name)
    plt.close()


def plot_trainv2(output, waveform, name):
    index = 1
    plt.figure(figsize=(8, 5), dpi=300)
    plt.subplot(211)
    plt.plot(output[:].cpu().detach().numpy())
    plt.plot(waveform[:].cpu().detach().numpy())
    plt.ylabel("waveform amplitude")

    plt.subplot(212)
    plt.plot(
        np.abs(output[:].cpu().detach().numpy() - waveform[:].cpu().detach().numpy())
    )
    plt.ylabel("residual amplitude")
    plt.xlabel("time[s*dt]")
    plt.savefig(name)
    plt.close()
