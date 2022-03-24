import numpy as np
import csv
import cumbersome_model
import complex_cnn
from scipy import signal
from scipy.fft import fft
#from sklearn.metrics import mean_squared_error, r2_score
import time
import torch
import os
import pickle
import matplotlib.pyplot as plt
import shutil
import random
#import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def read_train_data(file_name):
    with open(file_name, 'r', newline='') as f:
        lines = csv.reader(f)
        data = []
        for line in lines:
            data.append(line)

    data = np.array(data).astype(np.float)
    return data


def cut_data(file_name):
    with open(file_name, 'r', newline='') as f:
        lines = csv.reader(f)
        raw_data = []
        for line in lines:
            raw_data.append(line)
    raw_data = np.array(raw_data).astype(np.float)
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
        raw_data = np.array(raw_data).astype(np.float)
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
        print("GLUE DONE!" + filename2)


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
        print("The directory is deleted successfully")


def decode_data(data, std_num, mode=5):
    Channel_location = ["FP1", "FP2",
                        "F7", "F3", "FZ", "F4", "F8",
                        "T7", "C3", "CZ", "C4", "T8",
                        "P7", "P3", "PZ", "P4", "P8",
                        "O1", "O2"]
    model = cumbersome_model.UNet(n_channels=30, n_classes=30, bilinear=True)
    #model = complex_cnn.Complex_CNN(in_channels=1, out_channels=1, datanum=1024, bilinear=True)
    #model = model.cuda()
    resumeLoc = './UNet1D-real/final_RealEEG_5' + '/modelsave/BEST_checkpoint.pth.tar'
    checkpoint = torch.load(resumeLoc, map_location='cpu')
    #start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        # run the mdoel
        data = data[np.newaxis, :, :]
        data = torch.Tensor(data)
        #decode = model(data.cuda())
        decode = model(data)
        if int(std_num) != 0:
            decode = decode * std_num
    decode = np.array(decode.cpu()).astype(np.float)
    return decode


# model = tf.keras.models.load_model('./denoise_model/')

def main():
    for j in range(1):
        second1 = time.time()
        j = j+48
        name = "./UNet1D-real/resting_csv_test/" + str(j) + "_raw.csv"

        # -------------------cutting_data---------------------------
        try:
            os.mkdir("./temp2/")
        except OSError as e:
            print(e)

        #try:
        total = cut_data(name)
        #except:
        #print("jump out!")
        #continue
        # -------------------decode_data---------------------------
        for i in range(total):
            # file_name = './Real_EEG/test/Brain/510_{}.csv'.format(str(i))
            # data_clean = read_train_data(file_name)
            # torch_PSD(data_clean[0])

            file_name = './temp2/{}.csv'.format(str(i))
            data_noise = read_train_data(file_name)
            data_clean = data_noise
            # torch_PSD(data_noise[0])

            # print(data.shape, std)
            std = np.std(data_noise)
            avg = np.average(data_noise)
            '''
            if int(std) != 0:
                data_clean = np.array((data_clean - avg) / std).astype(np.float)
                data_noise = np.array((data_noise - avg) / std).astype(np.float)
            else:
                data_clean = np.array(data_clean - avg).astype(np.float)
                data_noise = np.array(data_noise - avg).astype(np.float)    
            '''

            # UNet
            d_data = decode_data(data_noise, std, 1)
            d_data = d_data[0]

            '''
            # EEGdenoiseNet
            for k in range(30):
                if k == 0:
                    tmp1 = np.expand_dims(data_noise[k], axis=0)
                    d_data = decode_data(tmp1, std, 5)
                    d_data = d_data[0]
                else:
                    tmp1 = np.expand_dims(data_noise[k], axis=0)
                    tmp2 = decode_data(tmp1, std, 5)
                    d_data = np.append(d_data, tmp2[0], axis=0)
                    #print("d_data2: ", d_data.shape, k)
    
            '''
            outputname = "./temp2/output{}.csv".format(str(i))
            save_data(d_data, outputname)
            #save_data(d_hidden, outputname)
            # save_data(d_hidden, hiddenname)
            #print(outputname, "OK")

            '''
            snr_mean, snr_std = SNR_cal(data_clean*std, data_noise*std)
            print("SNR(i&t):", snr_mean, snr_std)
            snr_mean, snr_std = SNR_cal(data_noise*std, d_data[0])
            print("SNR(i&d):", snr_mean, snr_std)
            '''
        # --------------------glue_data----------------------------
        outputname = './UNet1D-real/' + str(j) + "_output_sample.csv"
        #outputname = './final_result/1D_ResCNN/65ERP_out.csv'
        glue_data("./temp2/", total, outputname)
        # -------------------delete_data---------------------------
        dataDelete("./temp2/")
        second2 = time.time()

        print("decode time: ", second2 - second1)
        # plt.savefig(str(int(j)) + "Hz_Time_domain")
        # plt.close()

if __name__ == '__main__':
    main()