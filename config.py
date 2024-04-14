import numpy as np
import torch

#预训练语言模型的地址
PRE_MODEL_PATH = "/CSTemp/wqy/LLM/chinese_roberta_wwm_large/"

#文本最大词语长度
TEXT_LEN = 512

#文本的特征嵌入维度
INPUT_EMBEDDING_DIM = 1024

#卷积神经网络的输出嵌入维度
GCN_OUTPUT_DIM = 256




#bert模型的路径
BERT_MODEL_PATH = "/CSTemp/wqy/LLM/chinese_roberta_wwm_large/"

#
BERT_PAD_ID = 0

#医疗术语字典的位置
MEDICAL_VOCAB_PATH = "/CSTemp/wqy/pythonProject1/medical_vocab_normal.txt"


#
BATCH_SIZE = 256

DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'

LR = 0.00001

EPOCHES = 100

MODEL_SAVE_ROOT_PATH = "/CSTemp/wqy/pythonProject1/model_saved/"

REQUIRE_IMPROVE = 1000

CLASS_PATH = "/CSTemp/wqy/Paper55-journal/data/02-CMID/class.txt"
CLASS_LIST = [x.strip() for x in open(CLASS_PATH,encoding='utf-8').readlines()]

BEST_MODEL_PATH = ''


# 两阶段训练需要的参数
EPOCHS_TwoStage = 200

CLASS_NUM = 4



