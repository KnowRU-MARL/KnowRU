# #将数据输入模型、得到logits输出值
# import sys
# sys.path.insert(0,'./base_model')
import argparse
import torch
import torch.nn.functional as F

def get_logits(net, ob):
    base_model_logits = net(ob, model_original_out=True)

    return base_model_logits

#加载模型得到网络
def load_base_model(arglist):
    #得到基本网络
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_nets = []
    for i in arglist.base_model_name:
        str = arglist.base_model_dir + i
        # print(str)
        nets = torch.load(str).to(device)
        print(nets)
        base_nets.append(nets)


    return base_nets

def load_base_model2(arglist):
    #得到基本网络
    base_nets = []
    for i in arglist.base_model_name:

        nets = torch.load(i)

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

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str, default="base_model/", help="base_model路径")
    # parser.add_argument("--base_model_name", type=list, default=['a_c_0.pt','a_c_0.pt','a_c_1.pt','a_c_2.pt','a_c_1.pt'], help="base_model名字")
    parser.add_argument("--base_model_name", type=list,
                        default=['a_c_0.pt', 'a_c_1.pt', 'a_c_2.pt', 'a_c_3.pt', 'a_c_0.pt', 'a_c_1.pt', 'a_c_4.pt',
                                 'a_c_4.pt'], help="base_model名字")
    config = parser.parse_args()
    load_base_model(config)