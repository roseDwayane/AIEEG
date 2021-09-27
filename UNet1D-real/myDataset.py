import csv
import os
import random
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset
import pickle

epsilon = np.finfo(float).eps

class myDataset(Dataset):
    def __init__(self, mode, iter=20, data="0-3"):
        self.sample_rate = 256
        self.lenth = 78220
        self.lenthtest = 1024
        self.lenthval = 1024
        self.mode = mode
        self.iter = iter
        self.savedata = data

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        data_mode = ["Brain", "ChannelNoise", "Eye", "Heart", "LineNoise", "Muscle", "Other"]

        allFileList = os.listdir("./Real_EEG/train/Brain/")
        file_name = './Real_EEG/train/Brain/' + allFileList[idx]
        #print(file_name)
        data_clean = self.read_train_data(file_name)
        for i in range(7):
            file_name = './Real_EEG/train/' + data_mode[random.randint(0,6)] + '/' + allFileList[idx]
            if os.path.isfile(file_name):
                data_nosie = self.read_train_data(file_name)
                break
            else:
                data_nosie = data_clean + 10

        '''Read Real-EEG training data
                if self.mode == 2:
            file_name = './Real_EEG/train/Brain/' + str(idx) + '.csv'
            data = self.read_data((file_name))
            noise = np.zeros(1024)
        elif self.mode == 1:
            file_name = './val2/' + str(idx) + '.csv'
            data = self.read_train_data(file_name)
            noise = self.read_data("./eyes/eyemovement/" + str(random.randint(0, 209)) + ".csv")
        else:
            file_name = './train_data/' + str(idx%787) + '_' + str(idx%70) + '.csv'
            if os.path.isfile(file_name) == False:
                file_name = './train_data/' + str(idx % 787) + '_' + str(idx % 10) + '.csv'
            data = self.read_train_data(file_name)
            noise = self.read_data("./eyes/eyemovement/" + str(random.randint(0, 209)) + ".csv")

        '''

        #print("data_set", noise.shape)

        max_num = np.max(data_nosie)
        data_avg = np.average(data_nosie)
        data_std = np.std(data_nosie)
        #max_num = 100
        #print("max_num: ", max_num)

        #target = np.array(data / max_num).astype(np.float)
        if int(data_std) != 0:
            target = np.array((data_clean - data_avg) / data_std).astype(np.float)
            attr   = np.array((data_nosie - data_avg) / data_std).astype(np.float)
        else:
            target = np.array(data_clean - data_avg).astype(np.float)
            attr   = np.array(data_nosie - data_avg).astype(np.float)

        target = target.copy()
        target = torch.tensor(target, dtype=torch.float32)

        attr = attr.copy()
        attr = torch.tensor(attr, dtype=torch.float32)

        return attr, target, data_std

    def read_simulate_data(self, file_name):
        with open(file_name, 'rb+') as f:
            data = pickle.load(f)
        data = np.array(data).astype(np.float)
        return data

    def read_train_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float)

        row = np.array([0,1,2,3,4,5,6,12,13,14,15,16,22,23,24,25,26,27,29])
        new_data = []
        for i in range(19):
            #print(i, row[i])
            #print(data[row[i]].shape)
            new_data.append(data[row[i]])
        new_data = np.array(new_data).astype(np.float)
        #data = data.T
        return new_data

    def read_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float)
        #data = data.T
        return data

    def lowpass60hz(self, data, sample_rate):
        b, a = signal.butter(8, 2 * 60 / sample_rate, 'lowpass')  # 0.48 = 2*hz/sample_rate
        data = signal.filtfilt(b, a, data)  # data 要過濾的波
        return data

    def peak_gen(self, sample_rate):
        # peak --------------------------------
        x = random.randint(64, 512)
        while x % 2 == 0:
            x = random.randint(64, 512)
        y = random.uniform(1.5, 2.0)
        x1 = np.linspace(0, (x // 2 - 1), x // 2)
        x2 = np.linspace((x // 2), x, x // 2)
        y1 = y * x1
        y2 = -y * x2 + x * y
        peak = np.concatenate([y1, y2])
        X = np.linspace(0, 4, sample_rate)
        a = random.randint(1, sample_rate - x)
        peak = np.pad(peak, (a, sample_rate - x - a + 1), 'constant')
        # random noise ------------------------
        rand_arr = np.random.randn(sample_rate)
        # sin ---------------------------------
        x_sin = np.linspace(np.random.randn() * 10, np.random.randn(), sample_rate)
        sin_arr = np.sin(x_sin)

        noise = peak + 0.1 * rand_arr + 0.1 * sin_arr
        # plt.plot(X, noise)
        # plt.xlabel('Time [sec]')
        # plt.ylabel('Amplitude [µ]')
        # plt.show()
        return noise
