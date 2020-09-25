import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset, transform
from model import Transformer
from util import save_load as sl

def main():
    dataset = Dataset(transform=transform, n_datas=10000, seed=None) #生成10000个数据，确保字符都出现
    model = Transformer(n_head=4)
    try:
        trained_epoch = sl.find_last_checkpoint('./checkpoint')
        print('load model %d' % (trained_epoch))
    except Exception as e:
        print('no trained model found, {}'.format(e))
        return
    model = sl.load_model('./checkpoint', -1, model)
    model.eval()

    x, y, extra = dataset.__getitem__(0)
    # print(x.shape, y.shape)
    pred = model(torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)).squeeze()
    # print(pred.shape)
    # print(pred)
    pred = np.argmax(pred.detach().numpy(), axis=1)[1:]
    # print(pred)
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