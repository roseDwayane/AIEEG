import random
import csv
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import shutil

def B_rand_gen(L, M):
    '''
    if mode == 0:
        L, M = 0, 10
    elif mode == 1:
        L, M = 10, 30
    elif mode == 2:
        L, M = 30, 50

    '''
    freq = []
    amp = []
    for i in range(6):
        freq.append(random.uniform(L, M))
        amp.append(np.random.randn())

    time1 = np.linspace(np.random.randn(), np.pi * 2 * freq[0] * 4, 1024)
    time2 = np.linspace(np.random.randn(), np.pi * 2 * freq[1] * 4, 1024)
    time3 = np.linspace(np.random.randn(), np.pi * 2 * freq[2] * 4, 1024)
    time4 = np.linspace(np.random.randn(), np.pi * 2 * freq[3] * 4, 1024)
    time5 = np.linspace(np.random.randn(), np.pi * 2 * freq[4] * 4, 1024)
    time6 = np.linspace(np.random.randn(), np.pi * 2 * freq[5] * 4, 1024)
    data = []
    for i in range(19):
        data1 = np.sin(time1) * amp[0]
        data2 = np.sin(time2) * amp[1]
        data3 = np.sin(time3) * amp[2]
        data4 = np.sin(time4) * amp[3]
        data5 = np.sin(time5) * amp[4]
        data6 = np.sin(time6) * amp[5]
        data.append(data1 + data2 + data3 + data4 + data5 + data6)

    data = np.array(data).astype(np.float)
    return data, freq, amp

    # for reconstruct testing
    '''    
    new_data = data
    data_avg = np.average(new_data)
    data_std = np.std(new_data)
    target = np.array((new_data-data_avg) / data_std).astype(np.float)
    target = target.copy()
    print(freq)
    print(amp)
    return target, data_std, freq, amp
    '''

def dataRestore(name_log):
    with open(name_log, newline='') as f:
        rows = csv.reader(f, delimiter='\t')
        logger = []
        for row in rows:
            if row[0] == "ID":
                continue
            logger.append(row)

    para = np.array(logger).astype(np.float)

    time1 = np.linspace(np.random.randn(), np.pi * 2 * para[1] * 4, 1024)
    time2 = np.linspace(np.random.randn(), np.pi * 2 * para[2] * 4, 1024)
    time3 = np.linspace(np.random.randn(), np.pi * 2 * para[3] * 4, 1024)
    time4 = np.linspace(np.random.randn(), np.pi * 2 * para[4] * 4, 1024)
    time5 = np.linspace(np.random.randn(), np.pi * 2 * para[5] * 4, 1024)
    time6 = np.linspace(np.random.randn(), np.pi * 2 * para[6] * 4, 1024)
    data = []
    for i in range(19):
        data1 = np.sin(time1) * para[7]
        data2 = np.sin(time2) * para[8]
        data3 = np.sin(time3) * para[9]
        data4 = np.sin(time4) * para[10]
        data5 = np.sin(time5) * para[11]
        data6 = np.sin(time6) * para[12]
        data.append(data1 + data2 + data3 + data4 + data5 + data6)

    data = np.array(data).astype(np.float)
    return data

def dataGen(mode, L, H):
    if mode == "train":
        iter = 8192
    else:
        iter = 1024
    try:
        os.mkdir("./simulate_data/")
        os.mkdir("./datalog/")
    except OSError as e:
        print(e)
    try:
        os.mkdir("./simulate_data/" + mode + "/")
        os.mkdir("./datalog/" + mode + "/")
    except OSError as e:
        print(e)

    for i in range(iter):
        name = "./simulate_data/" + mode + "/" + str(i) + '.pk'
        name_log = "./datalog/" + mode + "/" + str(L) + "-" + str(H) + "log.csv"
        with open(name, 'wb+') as f:
            data, freq, amp = B_rand_gen(L, H)
            pickle.dump(data, f)
            print(i, " complete!")
        if os.path.isfile(name_log):
            logger = open(name_log, 'a')
            logger.write("\n%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
            i, freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], amp[0], amp[1], amp[2], amp[3], amp[4], amp[5]))
        else:
            logger = open(name_log, 'w')
            logger.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
            'ID', 'freq1', 'freq2', 'freq3', 'freq4', 'freq5', 'freq6', "amp1", "amp2", "amp3", "amp4", "amp5", "amp6"))
            logger.write("\n%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
            i, freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], amp[0], amp[1], amp[2], amp[3], amp[4], amp[5]))
    return 0

def dataDelete(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")

def dataInit(L, H):
    dataGen("train", L, H)
    dataGen("test", L, H)
    dataGen("val", L, H)