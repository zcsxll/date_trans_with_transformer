import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset, transform
from model import Transformer
from util import save_load as sl

def translate(model, x, y0): #这里只支持batch_size=1
    x_tensor = torch.from_numpy(x).unsqueeze(0) #(1, T, 37)， 37是输入数据词汇表长度
    enc_output = model.encoder(x_tensor) #编码部分可以先获取到
    y0_tensor = torch.from_numpy(y0).unsqueeze(0).unsqueeze(0) #(1, 1, 12)， 12是输出数据词汇表长度，解码最开始时只有开头的<pad>是已知的
    y_tensor = y0_tensor.clone()
    
    for step in range(14): #解码14次足以
        dec_output = model.decoder(enc_output, y_tensor) #以下三行是transformer的解码部分以及后面的模块
        pred = model.linear(dec_output)
        pred = torch.nn.Softmax(dim=-1)(pred) #this is very impotent

        y_tensor = torch.cat((y0_tensor, pred), dim=1) #解码得到的pred是除去开头<pad>字符的序列，下一次解码仍然需要<pad>
        # print(y_tensor.shape)
    return y_tensor.squeeze(0) #去除batch size

def main():
    dataset = Dataset(transform=transform, n_datas=10000, seed=None) #生成10000个数据，确保字符都出现
    model = Transformer(n_head=2)
    try:
        trained_epoch = sl.find_last_checkpoint('./checkpoint')
        print('load model %d' % (trained_epoch))
    except Exception as e:
        print('no trained model found, {}'.format(e))
        return
    model = sl.load_model('./checkpoint', -1, model)
    model.eval()

    x, y, extra = dataset.__getitem__(0) #值使用y的第0个特征向量，即<pad>对应的onehot向量
    # print(x.shape, y.shape)
    # pred = model(torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)).squeeze()
    pred = translate(model, x, y[0]) #日期格式转换时，对于输入序列，我们全部知道；但是对于输出序列，只有开头的<pad>的已知的
    # print(pred.shape)
    pred = np.argmax(pred.detach().numpy(), axis=1)[1:]
    # print(extra['machine_readable'])
    pred = ''.join([dataset.inv_machine_vocab[p] for p in pred])
    human_readable = extra['human_readable']
    machine_readable = extra['machine_readable']
    print('%s --> %s, answer: %s' % (human_readable, pred, machine_readable))
    
    # print(len(model.scores_for_paint), model.scores_for_paint[0].shape)
    # scores = np.array(model.scores_for_paint)
    # print(np.argmax(scores, axis=1))

    # f = plt.figure(figsize=(8, 8.5))
    # ax = f.add_subplot(1, 1, 1)
    # i = ax.imshow(scores, interpolation='nearest', cmap='Blues')

    # ax.set_xticks(range(30))
    # ax.set_xticklabels(human_readable[:30], rotation=0)

    # ax.set_yticks(range(10))
    # ax.set_yticklabels(machine_readable[:10], rotation=0)

    # plt.savefig('./attention.png')

if __name__ == '__main__':
    main()