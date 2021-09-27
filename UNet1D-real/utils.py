from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def draw_raw(filename, data, sub, snr):
    plt.subplot(3, 1, sub)
    X = np.linspace(-0.5, 0.5, len(data))
    plt.plot(X, data.cpu())
    plt.title(filename+" (snr:"+str(snr)+")")
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude [µ]')

def draw_psd(filename, data, window, snr_it, snr_id, sub1, sub2, order):
    plt.subplot(2, 1, sub1)
    zoom_in = int(len(data)/4)
    X = np.linspace(0, 1, zoom_in)
    if order == 1:
        plt.plot(X, data[:zoom_in].cpu(), color=(0, 0, 0), label='Input')
    if order == 2:
        plt.plot(X, data[:zoom_in].cpu(), color=(1, 0, 0), label='Decode')
    if order == 3:
        plt.plot(X, data[:zoom_in].cpu(), color=(0.3, 0.9, 0.8), label='target', ls = '--')
    plt.title(filename + "_raw (snr(input&target):" + str(snr_it) + ")")
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude [µ]')
    plt.grid(color='b', ls='--', lw = 0.2)
    #plt.legend()
    fs=256
    plt.subplot(2, 1, sub2)
    freq = torch.rfft(torch.tensor(data.cpu()), 1)
    freq = sum(np.abs(freq.T)) / len(freq.T)
    datalen = 50 * int(len(freq) / (fs / 2))
    x = np.linspace(0, 50, datalen)
    if order == 1:
        plt.plot(x, freq[:datalen], color=(0, 0, 0), label='Input')
    if order == 2:
        plt.plot(x, freq[:datalen], color=(1, 0, 0), label='Decode')
    if order == 3:
        plt.plot(x, freq[:datalen], color=(0.3, 0.9, 0.8), label='target', ls='--')
    plt.title(filename + "_PSD (snr(input&decode):" + str(snr_id) + ")")
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [Hz]')
    plt.grid(color='b', ls='--', lw=0.2)
    #plt.legend()
    # plt.show()

def imgSave(dir, file_name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.tight_layout()
    plt.savefig(dir + file_name)
    plt.clf()

def numpy_SNR(origianl_waveform, target_waveform):
    # 位 dB
    origianl_waveform = np.array(origianl_waveform.cpu()).astype(np.float)
    target_waveform = np.array(target_waveform.cpu()).astype(np.float)
    signal = np.sum(origianl_waveform ** 2)
    noise = np.sum((origianl_waveform - target_waveform) ** 2)
    snr = 10 * np.log10(signal / noise)
    if snr > 9999 or snr < -9999:
        snr = np.nan
    return snr

def SNR_cal(input, target, decode, max_num):
    snr_i_t = []
    snr_i_d = []
    for batch in range(input.shape[0]):
        temp_input = input * max_num[batch]

        temp_target = target * max_num[batch]
        temp_decode = decode * max_num[batch]
        temp_i_t = []
        temp_i_d = []
        for channel in range(input.shape[1]):
            temp_i_t.append(numpy_SNR(temp_input[batch][channel], temp_target[batch][channel]))
            temp_i_d.append(numpy_SNR(temp_input[batch][channel], temp_decode[batch][channel]))
        snr_i_t.append(temp_i_t)
        snr_i_d.append(temp_i_d)


    snr_i_d = np.array(snr_i_d)
    snr_i_t = np.array(snr_i_t)

    #print("(utils)SNR_id: ", snr_i_d)
    #print("(utils)SNR_it: ", snr_i_t)

    return snr_i_t, snr_i_d


