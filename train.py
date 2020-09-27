import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from dataset import Dataset, transform, collate_fn
from model import Transformer
from util import save_load as sl

def calc_accuracy(pred, answer):
    # print(pred)
    pred = np.argmax(pred, axis=2)
    answer = np.argmax(answer, axis=2)
    # print(pred.shape, answer.shape)
    correct = (pred == answer).astype(np.int)
    accuracy = correct.sum() / (pred.shape[0] * pred.shape[1])
    # print(accuracy)
    return accuracy

def train(model, loss_fn, optimizer, dataloader, epoch, use_gpu=False):
    pbar = tqdm(total=len(dataloader), bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
    pbar.set_description(f'Epoch %d' % epoch)

    for step, (batch_x, batch_y, _) in enumerate(dataloader):
        if use_gpu:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        pred = model(batch_x, batch_y[:, :-1, :])
        accuracy = calc_accuracy(pred.detach().cpu().numpy(), batch_y[:, 1:, :].detach().cpu().numpy())
        # loss = loss_fn(pred.transpose(1, 2), batch_y)
        loss = loss_fn(pred, batch_y[:, 1:, :])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(**{'loss':loss.detach().cpu().item(), 'accuracy':accuracy})
        pbar.update()
    sl.save_checkpoint('./checkpoint', epoch, model, optimizer)

    pbar.close()
    
def main(gpu_id=None):
    dataset = Dataset(transform=transform, n_datas=10000)
    pad_vec = np.zeros(len(dataset.human_vocab))
    pad_vec[dataset.human_vocab['<pad>']] = 1
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=6,
                                            shuffle=True,
                                            num_workers=6,
                                            collate_fn=partial(collate_fn, pad_vec))

    model = Transformer(n_head=2)
    if gpu_id is not None:
        print('use gpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        n_gpus = torch.cuda.device_count()
        # print('use %d gpu [%s]' % (n_gpus, gpu_id))
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[i for i in range(n_gpus)])
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    model = sl.load_model('./checkpoint', -1, model)
    optimizer = sl.load_optimizer('./checkpoint', -1, optimizer)

    try:
        trained_epoch = sl.find_last_checkpoint('./checkpoint')
        print('train form epoch %d' % (trained_epoch + 1))
    except Exception as e:
        print('train from the very begining, {}'.format(e))
        trained_epoch = -1
    for epoch in range(trained_epoch+1, 20):
        train(model, loss_fn, optimizer, dataloader, epoch, use_gpu=True if gpu_id is not None else False)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(gpu_id=None)
    else:
        main(gpu_id='0')
