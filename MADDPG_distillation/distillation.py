#将数据输入模型、得到logits输出值

import torch
from arguments import parse_args


def get_logits(net, ob, arglist):
    base_model_logits = net((ob.to(arglist.device, torch.float)), model_original_out=True)
    return base_model_logits

#加载模型得到网络
def load_base_model(arglist):
    #得到基本网络
    base_nets = []
    for i in arglist.base_model_name:

        nets = torch.load(arglist.base_model_dir + i)

        base_nets.append(nets)
    return base_nets

#outputs是预测值，labels是真实值
def softmax_cross_entropy_loss(outputs, labels):
    log_value = torch.log(outputs)
    clamp_value = log_value#lamp(log_value, 0.0, 1.0)
    cross_entropy = -torch.mean(labels * clamp_value)
    #是一个正数
    return cross_entropy

def self_softmax(logits, temperature =1):
    softmax_logits = torch.softmax(logits/float(temperature), dim = 1)
    return softmax_logits


if __name__ == '__main__':
    arglist = parse_args()
    a = load_base_model(arglist)
    b = get_logits(a[1],[-0.07871369, -0.10663601, -0.20697652, -0.8855329 ,  0.42468091,
        1.3611099 , -0.01701958,  1.23560257,  0.21951562, -0.05386585,
       -0.39050319,  0.99934424,  0.92417507,  0.63538094, -0.59278479,
        1.02761868, -0.71974493,  0.90375689, -0.54009181, -0.01095992,
        0.13635968,  0.74544278,  0.0316699 ,  1.09665844, -0.18397323,
        0.14456672],arglist)
    print(b)
