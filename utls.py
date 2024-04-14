
import datasets
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils import data
from torch.utils.data import DataLoader,Dataset
from config import *;
import jieba

#得到训练数据 验证数据 测试数据 的数据加载器
def get_loader():
    train_dataset = Dataset('train')
    val_dataset = Dataset('dev')
    test_dataset = Dataset('test')


    train_loader = data.DataLoader(train_dataset,batch_size=BATCH_SIZE)
    val_loader = data.DataLoader(val_dataset,batch_size=BATCH_SIZE)
    test_loader = data.DataLoader(test_dataset,batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader

# 将第二阶段的特征和标签组成数据集
import random
# 第二阶段的数据集       
import random
class generate_feature_dataset(data.Dataset):
    def __init__(self,data,target,cls_num_list,class_loss_list):
        self.data = data
        self.target = target
        self.class_dict = dict()   #类字典 存储每个类 样本对应的索引
        self.cls_num_list = cls_num_list
        self.cls_num = len(cls_num_list) #类别数
        self.cls_num_loss_list = class_loss_list


        self.cls_num_loss_list = [abs(x) for x in self.cls_num_loss_list]


        self.type = 'reverse+loss'
        #self.type = 'reverse+loss'
        for i in range(self.cls_num):
            #idx = torch.where(self.target == i)[0]
            idx = np.where(self.target == i)[0]
            self.class_dict[i] = idx



        #prob for reverse
        cls_num_list = np.array(self.cls_num_list)
        #每个类别出现的概率
        prob = list(cls_num_list / np.sum(cls_num_list))
        #列表逆序排列
        prob.reverse()
        self.prob = np.array(prob)

    def __len__(self):
        return len(self.target)

    def __getitem__(self,item):
        #如果类型是 平衡采样时 随机采样出一个类别和样本
        if self.type == 'balance':
            sample_class = random.randint(0,self.cls_num-1)
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)

        if self.type == 'reverse':
            sample_class = np.random.choice(range(self.cls_num), p=self.prob.ravel())
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)

        #考虑到样本数量和loss采样
        if self.type == 'reverse+loss':
            softmax_prob = softmax(self.prob)   #[0.15103812 0.15577206 0.1858594  0.17267454 0.18083011 0.15382577]

            cls_num_loss_list_array = np.array(self.cls_num_loss_list)   #[5398.52816355 3362.94646895 3271.02111852 1821.18226039 5176.45951819,5869.96457255]

            softmax_cls_num_loss_list = softmax_array(cls_num_loss_list_array)  #[nan nan nan nan nan nan]

            softmax_prob_p=[x*0.9for x in softmax_prob]

            softmax_cls_num_loss_list_p = [y*0.1 for y in softmax_cls_num_loss_list ]

            p = softmax_prob + softmax_cls_num_loss_list

            norm_p = list(p / np.sum(p))

            norm_p_array = np.array(norm_p)


            sample_class = np.random.choice(range(self.cls_num), p=norm_p_array.ravel())
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)


        temp_class = random.randint(0, self.cls_num - 1)
        temp_indexes = self.class_dict[temp_class]
        temp_index = random.choice(temp_indexes)
        item = temp_index

        data,target = self.data[item], self.target[item]
        data_dual,target_dual = self.data[sample_index], self.target[sample_index]

        return data,target,data_dual,target_dual

#对ndarray进行softmax
def softmax(x):
    """计算 softmax 激活函数"""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return softmax_x

def softmax_array(x):
    exp_x = np.exp(x - np.max(x))  # 避免指数溢出
    return exp_x / np.sum(exp_x)

import datasets
import torch
import torch.nn as nn
from torch.nn import init
from config import *

from torch.nn.utils.rnn import pad_sequence


#邻接矩阵的运算--余弦相似度
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_linjie_Bycosine(text_vectors_batches):
    batches = len(text_vectors_batches)
    # 根据相似性矩阵构造邻接矩阵，设置一个相似性阈值
    similarity_threshold = 0.3
    adjacency_matrix_batches = []
    for i in range(batches):
        cosine_sim_matrix = cosine_similarity(text_vectors_batches[i].cpu().numpy(), text_vectors_batches[i].cpu().numpy())
        adjacency_matrix = (cosine_sim_matrix > similarity_threshold).astype(int)
        adjacency_matrix_batches.append(adjacency_matrix)

    return np.array(adjacency_matrix_batches)



#加载 医疗术语词典
def load_word_list(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        return set(word.strip() for word in file.readlines())

#得到 医疗术语mask
def get_mask_text(text, word_list):
    words = list(jieba.cut(text))
    mask_text = [0]*TEXT_LEN

    #当前的词所在位置
    current_position = 0

    for i, word in enumerate(words):
        if word in word_list:
            current_word_length = len(word)
            #把 current_position到current_word_length+current_position的位置 都标为1
            for i in range(current_position, current_word_length+current_position):
                mask_text[i] = 1
            current_position = current_position + current_word_length
        else:
            current_word_length = len(word)
            current_position = current_position + current_word_length

    if len(mask_text) < TEXT_LEN:
        pad_len = TEXT_LEN - len(mask_text)
        mask_text += [0] * pad_len




    return  mask_text


from transformers import BertTokenizer
from torch.utils.data import DataLoader


#data加载
# class Dataset(datasets.Dataset):
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(Dataset).__init__()
        sample_path = data_path
        self.lines = open(sample_path, encoding='utf-8').readlines()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self,index):

        text = self.lines[index].split('\t')[0]
        label = self.lines[index].split('\t')[1].replace('\n', '')
        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']
        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        if type(label) == str:
            label_float = float(label)
            target = int(label_float)
        else:
            target = int(label)

        word_list = load_word_list(MEDICAL_VOCAB_PATH)
        medical_mask = get_mask_text(text, word_list)

        if len(medical_mask) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(medical_mask))
            medical_mask += [0] * pad_len


        return torch.tensor(input_ids[:TEXT_LEN]), torch.tensor(mask[:TEXT_LEN]), torch.tensor(target),torch.tensor(medical_mask[:TEXT_LEN])
        # print('index')
        # print(index)
        # text = []
        # label = []
        # medical_mask = []
        # mask_list = []
        # input_ids_list = []
        #
        # word_list = load_word_list(MEDICAL_VOCAB_PATH)
        # for i in index:
        #     text.append(self.lines[i].split('\t')[0])
        #     label.append(self.lines[i].split('\t')[1].replace('\n', ''))
        #
        #     # 分词
        #     medical_mask.append(get_mask_text(text[i], word_list))
        #
        #     tokened = self.tokenizer(text)
        #     input_ids = tokened['input_ids']
        #     mask = tokened['attention_mask']
        #
        #     if len(input_ids) < TEXT_LEN:
        #         pad_len = (TEXT_LEN - len(input_ids))
        #         # print(input_ids)
        #         # input_new = []
        #         # input_new.append(j for j in input_ids)
        #         # input_new.append(k for k in [[BERT_PAD_ID] * pad_len])
        #         input_ids = input_ids[0] + [BERT_PAD_ID] * pad_len
        #         # print(input_new)
        #         mask = mask[0] + [0] * pad_len
        #     mask_list.append(mask)
        #     input_ids_list.append(input_ids)
        #
        # target = [int(l) for l in label]
        #
        # tensor_list = [torch.tensor(lst) for lst in input_ids_list]
        # padded_tensor = pad_sequence(tensor_list, batch_first=True, padding_value=0)[:,:TEXT_LEN]
        # print(padded_tensor.shape)
        #
        # mask_tensor_list = [torch.tensor(lst) for lst in mask_list]
        # mask_padded_tensor = pad_sequence(mask_tensor_list, batch_first=True, padding_value=0)[:, :TEXT_LEN]
        #
        # print(mask_padded_tensor.shape)
        # print(torch.tensor(target).unsqueeze(-1).shape)
        # print(torch.tensor(medical_mask).shape)
        # return padded_tensor, mask_padded_tensor, torch.tensor(target).unsqueeze(-1), torch.tensor(medical_mask)


from torch.utils import data
from torch.utils.data import DataLoader

#得到训练数据 验证数据 测试数据 的数据加载器
def get_loader(train_path, dev_path, test_path):
    train_dataset = Dataset(train_path)
    val_dataset = Dataset(dev_path)
    test_dataset = Dataset(test_path)


    train_loader = data.DataLoader(train_dataset,batch_size=BATCH_SIZE)
    val_loader = data.DataLoader(val_dataset,batch_size=BATCH_SIZE)
    test_loader = data.DataLoader(test_dataset,batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader

#得到单个批次的数据加载器，用于得到数据特征和损失时用的
def get_loader_signal(train_path, dev_path, test_path):
    train_dataset = Dataset(train_path)
    val_dataset = Dataset(dev_path)
    test_dataset = Dataset(test_path)


    train_loader = data.DataLoader(train_dataset,batch_size=1)
    val_loader = data.DataLoader(val_dataset,batch_size=1)
    test_loader = data.DataLoader(test_dataset,batch_size=1)

    return train_loader, val_loader, test_loader



from sklearn import metrics
import torch.nn.functional as F
def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for b,(texts,mask,labels, medical_mask ) in enumerate(data_iter, 0):
            texts = texts.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)
            medical_mask = medical_mask.to(DEVICE)


            outputs = model(texts,mask,medical_mask)

            loss = F.cross_entropy(outputs, labels)

            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    # print('label_all:',labels_all)
    # print('predict_all:',predict_all)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=CLASS_LIST, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


#第二阶段的评估函数
def evaluate_twostage(feature_model, classifier_model,  data_iter, test=False):
    feature_model.eval()
    classifier_model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for b,(texts,mask,labels, medical_mask ) in enumerate(data_iter, 0):
            texts = texts.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)
            medical_mask = medical_mask.to(DEVICE)


            outputs_s = feature_model(texts,mask,medical_mask)
            outputs = classifier_model(outputs_s)

            loss = F.cross_entropy(outputs, labels)

            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    # print('label_all:',labels_all)
    # print('predict_all:',predict_all)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=CLASS_LIST, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)



def test(test_iter,model_path):
    # test
    #model = TextCNN()
    #model_path = MODEL_SAVE_ROOT_PATH + 'np03-best_loss.pth'
    model = torch.load(model_path)
    #model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

def test_twostage(feature_part, model_path, test_iter):
    # test
    #model = TextCNN()
    #model_path = MODEL_SAVE_ROOT_PATH + 'np03-best_loss.pth'
    class_model = torch.load(model_path)
    #model.load_state_dict(torch.load(BEST_MODEL_PATH))
    class_model.eval()
    feature_part.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate_twostage(feature_part,class_model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

#把数组中的0全部替换成-1
def replace_zeros_with_minus_one(arr):
    r = len(arr)
    c = len(arr[0])
    for i in range(r):
        for j in range(c):
            if arr[i][j] == 0:
                arr[i][j] = -1

def count_non_minus_one_elements(arr):
    count = 0
    for element in arr:
        if element != -1:
            count += 1
    return count





if __name__ == '__main__':
    #医疗术语mask的部分
    text = "今天的心情可真不错,糖尿病好治愈吗"
    word_list = load_word_list(MEDICAL_VOCAB_PATH)
    mask_text = get_mask_text(text,word_list)
    print(mask_text)
