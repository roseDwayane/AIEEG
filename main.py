import utils

if __name__ == '__main__':
    # parameter setting
    input_path = './sampledata/'
    input_name = 'sampledata.csv'
    sample_rate = 256 # input data sample rate
    modelname = 'ICUNet' # or 'UNetpp'
    output_path = './sampledata/'
    output_name = 'outputsample.csv'


    # step1: Data preprocessing
    total_file_num = utils.preprocessing(input_path+input_name, sample_rate)

    # step2: Signal reconstruction
    utils.reconstruct(modelname, total_file_num, output_path+output_name)