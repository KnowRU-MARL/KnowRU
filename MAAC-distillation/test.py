import torch.nn.functional as F
import torch

import numpy as np
import scipy.stats
def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)
# 随机生成两个离散型分布
x = [np.random.randint(1, 11) for i in range(10)]
print(x)
x = torch.tensor(x)
a = torch.sum(x)
print(a)
b = torch.mean(a.long())
print(a)
c = torch.mean(x)
print(c)
# print(np.sum(x))
# px = x / np.sum(x)
# print(px)
# y = [np.random.randint(1, 11) for i in range(10)]
# print(y)
# print(np.sum(y))
# py = y / np.sum(y)
# print(py)
#
#
# # 利用scipy API进行计算
# # scipy计算函数可以处理非归一化情况，因此这里使用
# # scipy.stats.entropy(x, y)或scipy.stats.entropy(px, py)均可
# KL = scipy.stats.entropy(x, y)
# print(KL)
#
# # 编程实现
# KL = 0.0
# for i in range(10):
#     KL += px[i] * np.log(px[i] / py[i])
#     # print(str(px[i]) + ' ' + str(py[i]) + ' ' + str(px[i] * np.log(px[i] / py[i])))
#
# print(KL)
# input=torch.randn(3,3)
# print(input)
# soft_input=torch.nn.Softmax(dim=1)
# a=soft_input(input)#对数据在1维上进行softmax计算
# print(a)
# b=torch.log(soft_input(input))#计算log
# print(b)
# loss=torch.nn.NLLLoss()
# target=torch.tensor([0,1,2])
# print(target)
# print('NLLLoss:')
#
# print(loss(b,target))#选取第一行取第0个元素，第二行取第1个，
# #---------------------第三行取第2个，去掉负号，求平均，得到损失值。
#
# loss=torch.nn.CrossEntropyLoss()
# print('CrossEntropyLoss:')
# print(loss(input,target))
# # p_logit: [batch, class_num]
# # q_logit: [batch, class_num]
