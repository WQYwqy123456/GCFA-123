import os, time, random, torch, warnings
import numpy as np
from utls import *
from modelTwoStage import *

warnings.filterwarnings("ignore")



def main():
    data_number = '02'
    train_path = ''
    dev_path = ''
    test_path = ''
    NUM_CLASSES = 0
    CLS_NUM_LIST = []


    if data_number == '01':
        train_path = "/CSTemp/wqy/pythonProject1/data/01-diabets/train.txt"
        dev_path = "/CSTemp/wqy/pythonProject1/data/01-diabets/dev.csv"
        test_path = "/CSTemp/wqy/pythonProject1/data/01-diabets/test.csv"
        NUM_CLASSES = 6
        CLS_NUM_LIST = [629, 1761, 1438, 1953, 717, 501]
    elif data_number == '02':
        train_path = "/CSTemp/wqy/Paper55-journal/data/02-CMID/new_train_data.csv"
        dev_path = "/CSTemp/wqy/Paper55-journal/data/02-CMID/new_dev_data.csv"
        test_path = "/CSTemp/wqy/Paper55-journal/data/02-CMID/new_test_data.csv"
        NUM_CLASSES = 4
        CLS_NUM_LIST = [2790, 1226, 944, 624]
    else:
        train_path = "/CSTemp/wqy/Paper55-journal/data/03-searchIntent/new_train_data2.csv"
        dev_path = "/CSTemp/wqy/Paper55-journal/data/03-searchIntent/new_dev_data2.csv"
        test_path = "/CSTemp/wqy/Paper55-journal/data/03-searchIntent/new_test_data2.csv"
        NUM_CLASSES = 10
        CLS_NUM_LIST = [337, 0, 187, 169, 155, 137, 85, 78, 72, 73]

    train_loader, dev_loader, test_loader = get_loader(train_path, dev_path, test_path)
    train_loader_signal, dev_loader_signal, test_loader_signal = get_loader_signal(train_path, dev_path, test_path)

    #初始化模型
    modeltwostage = ModelTwoStage(CLS_NUM_LIST,NUM_CLASSES)

    #训练
    modeltwostage.fit_oneStage(train_loader,dev_loader,test_loader)
    modeltwostage.fit_twoStage(train_loader_signal,dev_loader_signal,test_loader_signal)



if __name__ == '__main__':
    random_seeds = [1, 12, 123, 1234, 12345]

    for seed in random_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)

    main()
