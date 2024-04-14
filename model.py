import torch
import torch.nn as nn
import numpy as np
import time
from transformers import BertModel
from config import *
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


#模型框架结构
class Total_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Feature_Extractor()    #特征提取器bert
        self.attention_model = Medical_Attention()      #注意力模块
        self.gcn = GraphConvolution(INPUT_EMBEDDING_DIM,GCN_OUTPUT_DIM)
        self.pool = Mean_Pooling()
        self.classifier = Classifier()


    def forward(self,input,mask,medical_term_mask):

        out1 = self.feature_extractor(input,mask)
        #原始词语特征
        words_feature = out1.last_hidden_state
        #原始句子特征
        sentence_feature = out1.pooler_output

        atten_x = self.attention_model([words_feature,words_feature, words_feature],medical_term_mask)  #torch.Size([32, 20, 1024])

        linjie_matrix = torch.tensor(compute_linjie_Bycosine(words_feature))  #(32, 20, 20)

        #转化为矩阵 方便GCN
        # atten_x_matrix = atten_x.view( atten_x.shape(0)* atten_x.shape(1),  atten_x.shape(2))
        # linjie_matrix = linjie_matrix.view(linjie_matrix.shape(0)*linjie_matrix.shape(1),linjie_matrix.shape(2)).t()
        gcn_feature = self.gcn(linjie_matrix.to(DEVICE),atten_x.to(DEVICE))   #[torch.Size([20, 256])]   len=32

        gcn_feature = torch.stack(gcn_feature, dim=0)
        pool_feature = self.pool(gcn_feature)  #torch.Size([32, 256])

        # print(pool_feature.shape)
        # print(sentence_feature.shape)
        merge_feature = torch.cat([pool_feature, sentence_feature],dim=1)


        result = self.classifier(merge_feature)   #torch.Size([32, 6])

        return result

class Feature_part(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Feature_Extractor()    #特征提取器bert
        self.attention_model = Medical_Attention()      #注意力模块
        self.gcn = GraphConvolution(INPUT_EMBEDDING_DIM,GCN_OUTPUT_DIM)
        self.pool = Mean_Pooling()


    def forward(self,input,mask,medical_term_mask):

        out1 = self.feature_extractor(input,mask)
        #原始词语特征
        words_feature = out1.last_hidden_state
        #原始句子特征
        sentence_feature = out1.pooler_output

        atten_x = self.attention_model([words_feature,words_feature, words_feature],medical_term_mask)  #torch.Size([32, 20, 1024])

        linjie_matrix = torch.tensor(compute_linjie_Bycosine(words_feature))  #(32, 20, 20)

        #转化为矩阵 方便GCN
        # atten_x_matrix = atten_x.view( atten_x.shape(0)* atten_x.shape(1),  atten_x.shape(2))
        # linjie_matrix = linjie_matrix.view(linjie_matrix.shape(0)*linjie_matrix.shape(1),linjie_matrix.shape(2)).t()
        gcn_feature = self.gcn(linjie_matrix,atten_x)   #[torch.Size([20, 256])]   len=32

        gcn_feature = torch.stack(gcn_feature, dim=0)
        pool_feature = self.pool(gcn_feature)  #torch.Size([32, 256])

        # print(pool_feature.shape)
        # print(sentence_feature.shape)
        merge_feature = torch.cat([pool_feature, sentence_feature],dim=1)


        #result = self.classifier(merge_feature)   #torch.Size([32, 6])

        return merge_feature

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = Classifier()


    def forward(self,merge_feature):
        result = self.classifier(merge_feature)   #torch.Size([32, 6])
        return result





#模型结构中的特征提取模块
class Feature_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained(PRE_MODEL_PATH)
        for name,param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self,input,mask):
        out = self.model(input,mask)
        return out


#模型结构中的注意力模块
'''
agen_num: 代理向量的数量  默认为10
medical_term_mask: 医疗术语的mask列表 [len_token, 0 or 1]  0即mask即被掩码 1即非掩码
'''
class Medical_Attention(nn.Module):
    def __init__(self,agent_num=5,head_num=8,attn_drop=0.):
        super().__init__()
        self.agent_num = agent_num   #代理向量的个数
        self.head_num = head_num    #多头的个数, 默认为8
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)



    def forward(self,qkv,medical_term_mask):
        q, k, v = qkv[0], qkv[1], qkv[2]    #k=([32, 20, 1024])
        # a 是医疗术语token

        mask_q = q.clone()
        mask_q[medical_term_mask == 0] = 0   #掩掉非医疗术语的部分  torch.Size([32, 20, 1024, 1])

        #agent_token= F.avg_pool2d(mask_q.unsqueeze(3), (mask_q.unsqueeze(3).shape[2],mask_q.unsqueeze(3).shape[3])).squeeze(3)    #全局平均池化得到代理向量
        # 将第三个维度压缩，进行平均池化
        agent_token= torch.mean(mask_q, dim=1, keepdim=True)  #([32, 1, 1024]
        agent_token_multi = agent_token.repeat(1, self.agent_num, 1)  #([32, 5, 1024]


        b2 = nn.Parameter(torch.zeros(TEXT_LEN, self.agent_num)).to(DEVICE)
        b1 = nn.Parameter(torch.zeros(self.agent_num, TEXT_LEN)).to(DEVICE)
        trunc_normal_(b2, std=.02)
        trunc_normal_(b1, std=.02)

        # 转置矩阵，交换第一维和第二维
        k_T = k.transpose(1,2)   #torch.Size([32, 1024,20])
        #print('agent_token_multi')
        #print(agent_token_multi)
        #print('k_T')
        #print(k_T)
        #print('b1')
        #print(b1)
        agent_attn = self.softmax(agent_token_multi @ k_T + b1)  #torch.Size([32, 5, 20])
        agent_attn = self.attn_drop(agent_attn)   #torch.Size([32, 5, 20])
        agent_v = agent_attn @ v   #v=([32, 20, 1024])  agent_v=torch.Size([32, 5, 1024])

        q_attn = self.softmax(q @ agent_token_multi.transpose(-2,-1) + b2)  #q=([32, 20, 1024])  agent_token_multi=([32, 5, 1024]  q_atten=orch.Size([32, 20, 5])
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v   #q_attn=torch.Size([32, 20, 5])  agent_v = torch.Size([32, 5, 1024])  x=torch.Size([32, 20, 1024])
        # print(x.shape)
        return x



#模型结构中的图卷积网络模块
import torch
import torch.nn as nn
from utls import *


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：H*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化w

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        # init.kaiming_uniform_神经网络权重初始化，神经网络要优化一个非常复杂的非线性模型，而且基本没有全局最优解，
        # 初始化在其中扮演着非常重要的作用，尤其在没有BN等技术的早期，它直接影响模型能否收敛。

        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        #先判断输入特征和输入的邻接矩阵是否包含batches
        if input_feature.ndim >= 3:
            batch = input_feature.shape[0]
            batch_output = []
            for i in range(batch):
                #input_feature=[20,1024],self.weight=[1024,256]  support = [20,256]
                support = torch.mm(input_feature[i], self.weight)
                #adjacency=[20,20]
                output = torch.sparse.mm(adjacency[i].float().to(DEVICE), support.to(DEVICE))
                if self.use_bias:
                    output += self.bias
                batch_output.append(output)

            #batch_output_array = torch.tensor(np.array(batch_output.detach().numpy()))
            return batch_output
        else:
            # input_feature=[20,1024],self.weight=[1024,256]  support = [20,256]
            support = torch.mm(input_feature, self.weight)
            # adjacency=[20,20]
            output = torch.sparse.mm(adjacency, support)
            if self.use_bias:
                output += self.bias

            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'
#模型结构中的平均池化模块
class Mean_Pooling(nn.Module):


    def __init__(self):
        super().__init__()
    def forward(self,x):
        #对第二维进行平均池化操作
        average_pooled_tensor = torch.mean(x,dim=1)
        return average_pooled_tensor

#模型结构中的分类器模块
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_EMBEDDING_DIM+GCN_OUTPUT_DIM,CLASS_NUM)
    def forward(self,x):
        result = self.linear(x)
        return result


if __name__ == '__main__':
    input_sample = torch.randn(32,20,1024)    #模拟数据batch_size=32, 向量的编码维度为1024
    medical_term_mask = np.random.choice([0, 1], size=(32, 20))

    #测试注意力模块
    # medical_atten = Medical_Attention()
    # atten_r = medical_atten([input_sample,input_sample,input_sample],medical_term_mask)

    #测试邻接矩阵模块
    # linjie_matrix = compute_linjie_Bycosine(torch.randn(32,20,1024))
    # print('linjie_matrix.shape')
    # print(linjie_matrix.shape)

    #测试卷积网络模块
    atten_x = torch.randn([32, 20, 1024])
    #atten_x_matrix = atten_x.view(32 * 20, 1024)
    linjie_matrix = torch.randn([32, 20, 20])
    #linjie_matrix = linjie_matrix.view(32*20,20).t()
    gcn = GraphConvolution(1024,256)
    gcn_feature = gcn(linjie_matrix,atten_x)
    print('gcn_feature.shape')
    print(len(gcn_feature))   #32
    print(gcn_feature[0].shape)    #torch.Size([20, 256])
    print(gcn_feature[1])

    gcn_feature_tensor = torch.stack(gcn_feature)
    print(gcn_feature_tensor.shape)

    #测试平均池化模块
    mean_pool_feature = torch.mean(gcn_feature_tensor, dim=1)
    print(mean_pool_feature.shape)   #torch.Size([32, 256])

