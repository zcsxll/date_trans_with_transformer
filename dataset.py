import torch
import numpy as np
from faker import Faker
import random
from babel.dates import format_date
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

def transform(human_readable, machine_readable, human_vocab, machine_vocab):
    X = list(map(lambda x: human_vocab.get(x, '<unk>'), human_readable))
    Y = list(map(lambda x: machine_vocab.get(x, '<unk>'), machine_readable)) #len(Y) is always 10, because the format is YYYY-MM-DD
    '''
    我们输入当网络中的是[machine_vocab['<pad>']] + Y
    我们输出的目标是Y + [machine_vocab['<pad>']]
    因为第一次解码只使用了的最开始的<pad>（由mask控制），此时解码出来的字符应该是Y[0]
    最后一次解码时，使用了全部[machine_vocab['<pad>']] + Y，则目标输出就是Y + [machine_vocab['<pad>']]
    '''
    Y = [machine_vocab['<pad>']] + Y + [machine_vocab['<pad>']]
    # print(human_readable)
    # print(machine_readable)
    # print(X)
    # print(Y)
    def zcs(length, idx):
        ret = np.zeros(length)
        ret[idx] = 1
        return ret
    # Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Xoh = np.array(list(map(partial(zcs, len(human_vocab)), X)), dtype=np.float32)
    Yoh = np.array(list(map(partial(zcs, len(machine_vocab)), Y)), dtype=np.float32) #如果使用交叉熵loss，pytorch不需要label是onehot的；通过对比发现用MSELoss更好一些
    return Xoh, Yoh, {'human_readable':human_readable, 'machine_readable':machine_readable}
    # return Xoh, np.array(Y)

def collate_fn(pad_vec, batch):
    '''
    对于短的输入，onehost向量较少，需要补充<pad>对应的onehot向量，即pad_vec
    '''
    pad_vec = pad_vec.reshape(-1, 1)
    batch_x, batch_y, extra = zip(*batch)
    # print(batch_x.shape, batch_y.shape)
    max_len_x = max(x.shape[0] for x in batch_x)

    batch_x_paded = []
    for x in batch_x:
        pad_len = max_len_x - x.shape[0]
        pad = np.repeat(pad_vec, pad_len, axis=1).T
        batch_x_paded.append(np.vstack((x, pad)))
        # print(x.shape)

    batch_x = torch.FloatTensor(batch_x_paded)
    batch_y = torch.FloatTensor(batch_y)
    return batch_x, batch_y, extra
    # print(max_len_x, max_len_y)
    

# def collate_fn(batch):
#     batch_x, batch_y = zip(*batch)
#     batch_x = [torch.FloatTensor(x) for x in batch_x]
#     batch_y = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
#     return batch_x, batch_y

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, n_datas=10000, seed=12345):
        self.transform = transform

        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

        self.human_vocab = set()
        self.machine_vocab = set()
        self.dataset = []
        for i in tqdm(range(n_datas)):
            human_readable, machine_readable = self.load_date()
            self.dataset.append((human_readable, machine_readable))
            self.human_vocab.update(tuple(human_readable))
            self.machine_vocab.update(tuple(machine_readable))

        self.human_vocab = dict(zip(sorted(self.human_vocab) + ['<unk>', '<pad>'], list(range(len(self.human_vocab) + 2))))
        '''
        使用transformer时，需要给target序列也传给decoder，但是需要加mask，因为解码第4个输出时，只能使用前面0123这四个特征；后文有详解
        由于加mask后要进行softmax，因此至少得有一个有效元素
        但是解码第一个元素时，我们可供使用的target序列长度是0
        因此必须在最前边加一个<pad>
        '''
        self.inv_machine_vocab = dict(enumerate(sorted(self.machine_vocab) + ['<pad>']))
        self.machine_vocab = {v:k for k, v in self.inv_machine_vocab.items()}
        # print(self.human_vocab)
        # print(self.machine_vocab)
        # print(self.inv_machine_vocab)

    def __getitem__(self, idx):
        human_readable, machine_readable = self.dataset[idx] #dataset is a list of tuples
        return self.transform(human_readable, machine_readable, self.human_vocab, self.machine_vocab)

    def __len__(self):
        return len(self.dataset)

    def load_date(self):
        """
            Loads some fake dates 
            :returns: tuple containing human readable string, machine readable string, and date object
        """
        dt = self.fake.date_object()

        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US')
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()

        return human_readable, machine_readable

if __name__ == '__main__':
    dataset = Dataset(transform=transform, n_datas=1000)
    pad_vec = np.zeros(len(dataset.human_vocab))
    pad_vec[dataset.human_vocab['<pad>']] = 1
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=partial(collate_fn, pad_vec))

    for step, (batch_x, batch_y, extra) in enumerate(dataloader):
        print(step, batch_x.shape, batch_y.shape)
        # print(batch_x[0], extra[0]['human_readable'])
        print(batch_y[0], extra[0]['machine_readable']) #extra中的machine_readable部分长度是10，onehot向量是11个，其中第0个对应的是<pad>
        if step >= 0:
            break
