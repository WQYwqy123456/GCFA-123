import torch.optim
from transformers import BertModel
import torch.nn as nn
from utls import *
from config import *
from model import *


# 两阶段的训练方法类
import time
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from torch import functional as F

class ModelTwoStage():
    def __init__(self, cls_num_list, num_class):
        self.num_class = num_class
        self.cls_num_list = cls_num_list
        self.total_model = Total_Model()
        self.train_rule = None      #对于train_loader的数据采样策略
        self.usenorm = False        #是否使用归一化
        self.print_freq = 10        #打印频率 默认为10
        self.expansion_model = 'balance'   #采样策略
        self.device = DEVICE        #使用设备类型
        self.lr = LR                #学习率大小
        self.lr2 = 1e-5             #第二阶段学习率
        params = list(self.total_model.parameters())

        #第一阶段的优化器 ---使用随机梯度下降的优化算法
        self.optimizer_onestage = torch.optim.Adam(params,self.lr)
        self.criterion_onestage = nn.CrossEntropyLoss()

        #第二阶段的优化器----
        self.feature_part = Feature_part()
        self.classifier_part = Classifier()
        self.optimizer_twostage = torch.optim.SGD(self.classifier_part.parameters(),self.lr2)
        self.criterion_twostage = nn.CrossEntropyLoss(reduction='none')

    #第一阶段的训练，直到模型训练完毕
    def fit_oneStage(self,train_dataloader,dev_dataloader, test_dataloader):
        self.total_model = self.total_model.to(DEVICE)
        dev_best_loss = float('inf')
        dev_best_acc = 0
        total_batch = 0
        self.total_model.train()
        flag = False
        f = 0

        for e in range(EPOCHES):
            print('Epoch [{}/{}]'.format(e + 1, EPOCHES))
            for b, (input, mask, target, medical_mask) in enumerate(train_dataloader, 0):

                input = input.to(DEVICE)
                mask = mask.to(DEVICE)
                target = target.to(DEVICE)
                medical_mask = medical_mask.to(DEVICE)

                pred = self.total_model(input, mask, medical_mask)
                loss = self.criterion_onestage(pred, target)
                self.optimizer_onestage.zero_grad()
                loss.backward()
                self.optimizer_onestage.step()

                if total_batch % 100 == 0:
                    true = target.data.cpu()
                    predict = torch.argmax(pred, dim=1)
                    train_acc = metrics.accuracy_score(true, predict.cpu().numpy())
                    dev_acc, dev_loss = evaluate(self.total_model, dev_dataloader)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(self.total_model, MODEL_SAVE_ROOT_PATH + f'np03-best_loss_cmid.pth')
                        improve = '*'
                        last_improve = total_batch

                    if dev_acc > dev_best_acc:
                        dev_best_acc = dev_acc
                        torch.save(self.total_model, MODEL_SAVE_ROOT_PATH + f'np03-_best_acc_cmid.pth')
                        improve = '*'
                    else:
                        improve = ''
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},{5}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))
                total_batch += 1
                if total_batch - last_improve > REQUIRE_IMPROVE:
                    # ▒~L▒~A▒~[~Floss▒~E▒~G1000batch没▒~K▒~Y~M▒~L▒~S▒~]~_训▒~C
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
            test_model_path =  MODEL_SAVE_ROOT_PATH + f'np03-best_loss_cmid.pth'
            test(test_dataloader,test_model_path)

    #第二阶段的训练
    def fit_twoStage(self,train_dataloader, dev_dataloader, test_dataloader):
        #stage one: 模型准备
        print("第二阶段的模型准备开始******************************************")
        total_model = self.total_model
        model_state_dict = self.total_model.state_dict()
        feature_state_dict = self.feature_part.state_dict()
        classifier_state_dict = self.classifier_part.state_dict()

        for name, param in model_state_dict.items():
            if name in feature_state_dict:
                feature_state_dict[name].copy_(param)
            if name in classifier_state_dict:
                classifier_state_dict[name].copy_(param)

        self.feature_part.load_state_dict(feature_state_dict)
        self.classifier_part.load_state_dict(classifier_state_dict)
        self.feature_part = self.feature_part.to(self.device)
        self.classifier_part = self.classifier_part.to(self.device)

        print("第二阶段的模型准备完毕******************************************")

        print("第二阶段数据准备开始******************************************")

        #遍历拿到数据特征
        train_feature_dataset,train_feature_dataloader = self.get_features(self.feature_part, self.classifier_part, train_dataloader)
        #验证集保持正常
        # dev_dataloader
        #dev_feature_dataset, dev_feature_dataloader = self.get_features(self.feature_part,self.classifier_part,dev_dataloader)
        #测试集保持正常
        # test_dataloader
        test_feature_dataset, test_feature_dataloader = self.get_features(self.feature_part, self.classifier_part, test_dataloader)
        print("第二阶段数据准备完成******************************************")

        print("第二阶段分类器微调开始******************************************")
        MIX_feature = []   #混合特征
        MIX_label = []     #混合样本原始标签
        MIX_pre_label = []  #混合样本预测标签
        ORIGIN_label = []   #原始样本原始标签
        ORIGIN_feature = [] #原始样本特征
        ORIGIN_pre_label= [] #原始样本预测标签
        cls_num_list = np.array(self.cls_num_list)
        total_batch = 0
        dev_best_loss = float('inf')
        dev_best_acc = 0
        total_batch = 0
        self.classifier_part.train()
        flag = False
        f = 0

        for epoch in range(EPOCHS_TwoStage):
            losses = []
            for i, (inputs, targets, inputs_dual, targets_dual) in enumerate(train_feature_dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                inputs_dual = inputs_dual.to(self.device)
                targets_dual = targets_dual.to(self.device)

                num_batch = len(targets)

                lam = cls_num_list[targets.cpu().data] / (
                            cls_num_list[targets.cpu().data] + cls_num_list[targets_dual.cpu().data])
                lam = torch.tensor(lam, dtype=torch.float).view(num_batch, -1).to(self.device)

                if self.expansion_model == 'balance':
                    lam = 0.5 * torch.ones_like(lam)
                elif self.expansion_model == 'reverse':
                    lam = 1 - lam

                #生成混合样本
                mix = lam * inputs + (1 - lam) * inputs_dual

                outputs_o = self.classifier_part(inputs)  # 把原始input扔给classifer
                outputs_s = self.classifier_part(mix)  # 把混合样本扔给classifier
                _, pred_o = torch.max(outputs_o, 1)
                _, pred_s = torch.max(outputs_s, 1)



                # 存储混合样本特征、混合样本原始标签、混合样本预测标签, 原始样本特征、原始样本标签、预测样本标签
                # MIX_feature += mix
                # MIX_label += targets_dual
                # MIX_pre_label += pred_s
                # ORIGIN_feature += inputs
                # ORIGIN_label += targets
                # ORIGIN_pre_label += pred_o

                #计算混合样本损失
                loss_o = self.criterion_twostage(outputs_o, targets)
                loss_s = 0.5 * self.criterion_twostage(outputs_s, targets) + 0.5 * self.criterion_twostage(outputs_s, targets_dual)

                loss = loss_o + loss_s  # 计算混合损失  tensor([746.4760, 772.1915, 505.2329, 944.1188, 944.1188, 398.7255],device='cuda:3', grad_fn=<AddBackward0>)
                loss_sum = loss.sum()

                #损失反馈传播、修改模型参数
                self.optimizer_twostage.zero_grad()
                loss_sum.backward()
                self.optimizer_twostage.step()



                #每隔一百轮、进行模型验证，如果比之前好就保存
                if total_batch % 100 == 0:
                    true = targets.data.cpu()
                    predict = torch.argmax(outputs_o, dim=1)
                    train_acc = metrics.accuracy_score(true, predict.cpu().numpy())
                    dev_acc, dev_loss = evaluate_twostage(self.feature_part, self.classifier_part, dev_dataloader)

                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(self.classifier_part, MODEL_SAVE_ROOT_PATH + f'np03-best_loss_class_part_cmid.pth')
                        torch.save(self.classifier_part.state_dict(),
                                   MODEL_SAVE_ROOT_PATH + f'np03_best_loss_class_part_param_cmid.pth')  # 保存模型参数
                        improve = '*'
                        last_improve = total_batch

                    if dev_acc > dev_best_acc:
                        dev_best_acc = dev_acc
                        torch.save(self.classifier_part, MODEL_SAVE_ROOT_PATH + f'np03-_best_acc_class_part_cmid.pth')
                        torch.save(self.classifier_part.state_dict(),
                                   MODEL_SAVE_ROOT_PATH + f'np03-_best_acc_class_part_param_cmid.pth')
                        improve = '*'
                    else:
                        improve = ''
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},{5}'
                    print(msg.format(total_batch, loss_sum.item(), train_acc, dev_loss, dev_acc, improve))
                total_batch += 1
                if total_batch - last_improve > REQUIRE_IMPROVE:
                    # ▒~L▒~A▒~[~Floss▒~E▒~G1000batch没▒~K▒~Y~M▒~L▒~S▒~]~_训▒~C
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                print('第二阶段分类器微调结束******************************************')
                break
        print('第二阶段的模型测试开始******************************************')
        test_model_path = MODEL_SAVE_ROOT_PATH + f'np03-best_loss_class_part_cmid.pth'
        test_twostage(self.feature_part, test_model_path, test_dataloader)
        print('第二阶段的模型结束******************************************')


        # 遍历拿到数据特征
    def get_features(self, feature_part, class_part, dataloader):
        mem_features, mem_targets = [], []
        true_target = []

        mem_losses = np.zeros((self.num_class, np.sum(self.cls_num_list)))
        replace_zeros_with_minus_one(mem_losses)   #将元素0都替换成-1
        index = 0
        for i, (inputs, masks, targets, medical_mask) in enumerate(dataloader):
            inputs = inputs.to(self.device)    #torch([1,1280])
            masks = masks.to(self.device)
            targets = targets.to(self.device)    #tensor([0], device='cuda:3')
            medical_mask = medical_mask.to(self.device)

  
            features = feature_part(inputs, masks,medical_mask)   #tensor([[-1.6350e+02,  5.1987e+02, -1.3485e+01,  ...,  7.5146e-01,-6.2012e-01,  3.5185e-01]], device='cuda:3', grad_fn=<CatBackward0>)


            outputs = class_part(features)   #tensor([[ 248.4710,  284.1881, -236.9037,   55.8283, -169.7907,  245.8788]],device='cuda:3', grad_fn=<AddmmBackward0>)
            predict = torch.argmax(outputs, dim=1)
            outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)


            losses = self.criterion_onestage(outputs_softmax, targets)
            mem_features.extend(np.array(features.cpu().data))   #吧获得的特征存起来
            mem_targets.extend(np.array(outputs.cpu().data))     #吧获得的预测标签也存起来
            true_target.extend(np.array(targets.cpu().data))

            #根据真实标签进行归类
            current_index = count_non_minus_one_elements(mem_losses[targets.cpu().numpy()][0])

            mem_losses[int(targets.cpu().item())][current_index] = losses.item()
  
        mem_outputs = np.array(mem_features)  # 第一阶段的特征
        mem_targets = np.array(mem_targets)   # 第一阶段该样本的预测标签
        mem_losses = np.array(mem_losses)     # 第一阶段该样本带来的损失
        true_targets = np.array(true_target)
            
        #对每个key对应的value值求和 dict
        class_loss_sum = {}
        for key in range(self.num_class):
            class_loss_sum[key] = sum(mem_losses[key])

  
        #对每个key对应的value值求和 list
        sorted_keys = sorted(class_loss_sum.keys())
        sorted_values = [class_loss_sum[key] for key in sorted_keys]

  
        #生成特征数据集
        #类别损失data,target,cls_num_list,class_loss_list)
        dataset = generate_feature_dataset(torch.FloatTensor(mem_outputs), torch.from_numpy(true_targets), self.cls_num_list, sorted_values )
  
  
  
  
        dataloader =  DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
  
        return dataset, dataloader



if __name__ == '__main__':
    modelteostage = ModelTwoStage()



































