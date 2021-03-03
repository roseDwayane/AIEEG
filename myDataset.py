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
        self.lenth = 8192
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
                if self.mode == 2:
            file_name = './VEPdata/' + str(idx) + '.csv'
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
        if self.mode == 2:
            file_name = self.savedata + 'val/' + str(idx) + '.pk'
            data = self.read_simulate_data(file_name)
        elif self.mode == 1:
            file_name = self.savedata + 'test/' + str(idx) + '.pk'
            data = self.read_simulate_data(file_name)
        else:
            file_name = self.savedata + 'train/' + str(idx) + '.pk'
            data = self.read_simulate_data(file_name)

        noise = np.zeros(1024)
        #data = self.lowpass60hz(data, self.sample_rate)

        #data = self.B_rand_gen()

        blink_para = [1, 1,
                      .7, .7, .7, .7, .7,
                      .5, .5, .5, .5, .5,
                      .3, .3, .3, .3, .3,
                      .2, .2, .2, .2, .2,
                      .1, .1, .1, .1, .1,
                      .01, .01, .01]

        #print("data_set", noise.shape)

        max_num = np.max(data)
        data_avg = np.average(data)
        data_std = np.std(data)
        #max_num = 100
        #print("max_num: ", max_num)

        #target = np.array(data / max_num).astype(np.float)
        target = np.array((data-data_avg) / data_std).astype(np.float)
        target = target.copy()

        attr = np.zeros((19, 1024))
        for i in range(19):
            attr[i] = np.array((data[i] + blink_para[i] * noise[0] - data_avg) / data_std).astype(np.float)
            #attr[i] = np.array((data[i] + blink_para[i] * noise[0]) / max_num).astype(np.float)
        attr = attr.copy()

        target = torch.Tensor(target)
        attr = torch.Tensor(attr)

        #return attr, target, max_num
        return attr, target, data_std

    def A_hz_gen(self, idx=0):
        time = []
        for i in range(self.iter):
            temp = np.linspace(0, np.pi * 2 * i, 512)
            time.append((temp))
        data = []
        for i in range(19):
            temp = []
            for j in range(self.iter):
                temp2 = np.sin(time[j])
                temp.append(temp2)

            # rand_arr = np.random.randn(512)
            data_sum = 0
            for j in range(self.iter):
                data_sum = data_sum + temp[j]
            data.append(data_sum)

        data = np.array(data)
        data = data.astype(np.float)
        return data

    def B_rand_gen(self, mode):
        if mode == 0:
            L, M = 0, 30
        elif mode == 1:
            L, M = 30, 33
        elif mode == 2:
            L, M = 33, 36
        time1 = np.linspace(np.random.randn(), np.pi * 2 * random.uniform(L, M) * 4, 1024)
        time2 = np.linspace(np.random.randn(), np.pi * 2 * random.uniform(L, M) * 4, 1024)
        time3 = np.linspace(np.random.randn(), np.pi * 2 * random.uniform(L, M) * 4, 1024)
        time4 = np.linspace(np.random.randn(), np.pi * 2 * random.uniform(L, M) * 4, 1024)
        time5 = np.linspace(np.random.randn(), np.pi * 2 * random.uniform(L, M) * 4, 1024)
        time6 = np.linspace(np.random.randn(), np.pi * 2 * random.uniform(L, M) * 4, 1024)
        data = []
        for i in range(19):
            data1 = np.sin(time1) * np.random.randn()
            data2 = np.sin(time2) * np.random.randn()
            data3 = np.sin(time3) * np.random.randn()
            data4 = np.sin(time4) * np.random.randn()
            data5 = np.sin(time5) * np.random.randn()
            data6 = np.sin(time6) * np.random.randn()
            data.append(data1+data2+data3+data4+data5+data6)

        data = np.array(data)
        data = data.astype(np.float)
        return data

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
