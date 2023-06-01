import numpy as np
import csv
from model import cumbersome_model2
from model import UNet_family
import time
import torch
import os
import random
import shutil
from scipy.signal import decimate, resample_poly, firwin, lfilter


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def resample(signal, fs):
    # downsample the signal to a sample rate of 256 Hz
    if fs>256:
        fs_down = 256 # Desired sample rate
        q = int(fs / fs_down) # Downsampling factor
        signal_new = []
        for ch in signal:
            x_down = decimate(ch, q)
            signal_new.append(x_down)

    # upsample the signal to a sample rate of 256 Hz
    elif fs<256:
        fs_up = 256  # Desired sample rate
        p = int(fs_up / fs)  # Upsampling factor 
        signal_new = []
        for ch in signal:
            x_up = resample_poly(ch, p, 1)
            signal_new.append(x_up)

    else:
        signal_new = signal

    signal_new = np.array(signal_new).astype(np.float64)

    return signal_new

def FIR_filter(signal, lowcut, highcut):
    fs = 256.0
    # Number of FIR filter taps
    numtaps = 1000
    # Use firwin to create a bandpass FIR filter
    fir_coeff = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    # Apply the filter to signal:
    filtered_signal  = lfilter(fir_coeff, 1.0, signal)
    
    return filtered_signal


def read_train_data(file_name):
    with open(file_name, 'r', newline='') as f:
        lines = csv.reader(f)
        data = []
        for line in lines:
            data.append(line)

    data = np.array(data).astype(np.float64)
    return data


def cut_data(raw_data):
    raw_data = np.array(raw_data).astype(np.float64)
    total = int(len(raw_data[0]) / 1024)
    for i in range(total):
        table = raw_data[:, i * 1024:(i + 1) * 1024]
        filename = './temp2/' + str(i) + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table)
    return total


def glue_data(file_name, total, output):
    gluedata = 0
    for i in range(total):
        file_name1 = file_name + 'output{}.csv'.format(str(i))
        with open(file_name1, 'r', newline='') as f:
            lines = csv.reader(f)
            raw_data = []
            for line in lines:
                raw_data.append(line)
        raw_data = np.array(raw_data).astype(np.float64)
        #print(i)
        if i == 0:
            gluedata = raw_data
        else:
            smooth = (gluedata[:, -1] + raw_data[:, 1]) / 2
            gluedata[:, -1] = smooth
            raw_data[:, 1] = smooth
            gluedata = np.append(gluedata, raw_data, axis=1)
    #print(gluedata.shape)
    filename2 = output
    with open(filename2, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(gluedata)
        #print("GLUE DONE!" + filename2)


def save_data(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def dataDelete(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(e)
    else:
        pass
        #print("The directory is deleted successfully")


def decode_data(data, std_num, mode=5):
    
    if mode == "ICUNet":
        model = cumbersome_model2.UNet1(n_channels=30, n_classes=30)
        resumeLoc = './model/ICUNet/modelsave' + '/checkpoint.pth.tar'
      
    elif mode == "UNetpp":
        model = UNet_family.NestedUNet3(num_classes=30)
        resumeLoc = './model/UNetpp/modelsave' + '/checkpoint.pth.tar'
     
    #model = complex_cnn.Complex_CNN(in_channels=1, out_channels=1, datanum=1024, bilinear=True)
    #model = model.cuda()
    #resumeLoc = './1103_1_RealEEG_1' + '/modelsave/checkpoint.pth.tar'
    
    checkpoint = torch.load(resumeLoc, map_location='cpu')
    #start_epoch = checkpoint['epoch']
    #print(checkpoint)
    model.load_state_dict(checkpoint['state_dict'],False)
    model.eval()
    with torch.no_grad():
        # run the mdoel
        data = data[np.newaxis, :, :]
        data = torch.Tensor(data)
        #decode = model(data.cuda())
        #decode = model(data)
        if mode == "UNetpp" or mode == "UNetpp_block" or mode == "Trans" or mode == "Trans_block":
            decode1, decode2, decode = model(data)
        else:
            decode = model(data)
        if int(std_num) != 0:
            decode = decode * std_num
    decode = np.array(decode.cpu()).astype(np.float64)
    return decode

def preprocessing(filename, samplerate):
    # establish temp folder
    try:
        os.mkdir("./temp2/")
    except OSError as e:
        dataDelete("./temp2/")
        os.mkdir("./temp2/")
        print(e)
    
    # read data
    signal = read_train_data(filename)
    #print(signal.shape)
    # resample
    signal = resample(signal, samplerate)
    #print(signal.shape)
    # FIR_filter
    signal = FIR_filter(signal, 1, 50)
    #print(signal.shape)
    # cutting data
    total_file_num = cut_data(signal)

    return total_file_num


# model = tf.keras.models.load_model('./denoise_model/')
def reconstruct(model_name, total, outputfile):
    # -------------------decode_data---------------------------
    second1 = time.time()
    for i in range(total):
        file_name = './temp2/{}.csv'.format(str(i))
        data_noise = read_train_data(file_name)
        
        std = np.std(data_noise)
        avg = np.average(data_noise)

        data_noise = (data_noise-avg)/std

        # UNet
        d_data = decode_data(data_noise, std, model_name)
        d_data = d_data[0]

        outputname = "./temp2/output{}.csv".format(str(i))
        save_data(d_data, outputname)

    # --------------------glue_data----------------------------
    #outputname = "D:/resting_test/" + nname + "/" + trainset + "/" + str(j) + ".csv"
    #outputname = "D:/resting_test/block_result/" + nname + "/" + str(block_ch) + "/" + str(j) + ".csv"
    #outputname = "./sampledata/outputsample.csv"
    glue_data("./temp2/", total, outputfile)
    # -------------------delete_data---------------------------
    dataDelete("./temp2/")
    second2 = time.time()

    print("Reconstruct ", outputfile, " has been success ", second2 - second1, "sec(s)")

