#coding=utf-8
import argparse
import os
import time
import logging
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

cudnn.benchmark = True

import numpy as np
import models

from data import datasets
from data.sampler import CycleSampler
from data.data_utils import init_fn
from utils import Parser,criterions

from predict import AverageMeter
import setproctitle  # pip install setproctitle
from visdom import Visdom


parser = argparse.ArgumentParser()

parser.add_argument('-cfg', '--cfg', default='1_EESPNet_16x_PRelu_GDL_all', required=True, type=str,
                    help='Your detailed configuration of the network')
parser.add_argument('-gpu', '--gpu', default='0', type=str, required=True,
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('-batch_size', '--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('-restore', '--restore', default='model_last.pth', type=str)# model_last.pth

path = os.path.dirname(__file__)

## parse arguments  对象 参数 字符串
args = parser.parse_args()    #parser增加的属性都会在args实例中
args = Parser(args.cfg, log='train').add_args(args)   ##get sth from DMFNet_GDL_all.yaml
# arg = DMFNet_GDL_all.yaml(cfg) + input
# args.net_params.device_ids= [int(x) for x in (args.gpu).split(',')]
ckpts = args.makedir()      #创建ckpts目录

args.resume = os.path.join(ckpts,args.restore)  ### specify the epoch ->  model_last.pth or sth

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)  #set random seed to compare and make better

    Network = getattr(models, args.net)  ##really  can call DMFNet in DMFNet_16x.py??
    model = Network(**args.net_params)   #dict
    model = torch.nn.DataParallel(model).cuda()

    optimizer = getattr(torch.optim, args.opt)(model.parameters(), **args.opt_params)
    criterion = getattr(criterions, args.criterion)  #getattr makes functional  GeneralizedDiceLoss


    msg = ''  #massage
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
    else:
        msg = '-------------- New training session ----------------'

    msg += '\n' + str(args)
    logging.info(msg)

    # Data loading code
    Dataset = getattr(datasets, args.dataset) #make Dataset become BraTSDataset

    train_list = os.path.join(args.train_data_dir, args.train_list)  ##all.txt
    train_set = Dataset(train_list, root=args.train_data_dir, for_train=True,transforms=args.train_transforms)

    num_iters = args.num_iters or (len(train_set) * args.num_epochs) // args.batch_size
    num_iters -= args.start_iter
    train_sampler = CycleSampler(len(train_set), num_iters*args.batch_size)  ##???
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collate,##
        sampler=train_sampler,  ##
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn)

    start = time.time()

    enum_batches = len(train_set)/ float(args.batch_size) # nums_batch per epoch or iterations per epoch
    # len(train_set) = num of data pictures

    losses = AverageMeter()
    torch.set_grad_enabled(True)

    # visdom
    sum = 0
    viz = Visdom(port=2333)
    viz.line([1.], [0.], win="train_loss_epo",
             opts=dict(title='train loss epo', xlabel='epoch', ylabel='train_loss_epo'))

    for i, data in enumerate(train_loader, args.start_iter):

        elapsed_bsize = int( i / enum_batches)+1   #nums of batchs that elapsed   (maybe per epoch)
        epoch = int((i + 1) / enum_batches)
        setproctitle.setproctitle("Epoch:{}/{}".format(elapsed_bsize,args.num_epochs)) #set name of proceeding

        # actual training
        adjust_learning_rate(optimizer, epoch, args.num_epochs, args.opt_params.lr)

        data = [t.cuda(non_blocking=True) for t in data]
        x, target = data[:2]

        output = model(x)

        if not args.weight_type: # compatible for the old version
            args.weight_type = 'square'

        if args.criterion_kwargs is not None: #not judge content but is None judges Whether the list or dict is declared and defined
            loss = criterion(output, target, **args.criterion_kwargs)
        else:
            loss = criterion(output, target)   #GeneralizedDiceLoss   loss1 loss2 loss4

        # measure accuracy and record loss
        losses.update(loss.item(), target.numel())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # visdom
        l = int(enum_batches) + 1
        sum += losses.avg
        if (i + 1) % l == 0:
            mean_loss = sum / l
            sum = 0
            viz.line([mean_loss], [epoch], win='train_loss_epo', update='append')

        # save model
        if (i+1) % int(enum_batches * args.save_freq) == 0 \
            or (i+1) % int(enum_batches * (args.num_epochs -1))==0\
            or (i+1) % int(enum_batches * (500 -0))==0\
            or (i+1) % int(enum_batches * (900 -0))==0\
            or (i+1) % int(enum_batches * (500 -1))==0\
            or (i+1) % int(enum_batches * (900 -1))==0\
            or (i+1) % int(enum_batches * (args.num_epochs -2))==0\
            or (i+1) % int(enum_batches * (500 -2))==0\
            or (i+1) % int(enum_batches * (900 -2))==0\
            or (i+1) % int(enum_batches * (args.num_epochs -3))==0\
            or (i+1) % int(enum_batches * (500 -3))==0\
            or (i+1) % int(enum_batches * (900 -3))==0\
            or (i+1) % int(enum_batches * (args.num_epochs -4))==0\
            or (i+1) % int(enum_batches * (500 -4))==0\
            or (i+1) % int(enum_batches * (900 -4))==0:

            file_name = os.path.join(ckpts, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'iter': i,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)



        msg = 'Iter {0:}, Epoch {1:.4f}, Loss {2:.7f}'.format(i+1, (i+1)/enum_batches, losses.avg)
        logging.info(msg)

        losses.reset()

    # time and save model
    i = num_iters + args.start_iter
    file_name = os.path.join(ckpts, 'model_last.pth')
    torch.save({
        'iter': i,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        },
        file_name)

    msg = 'total time: {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        param_group['lr'] = lr
    # print('lr:',lr)
""" or   warm up
def adjust_learning_rate(optimizer, epoch):
    #Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = args.lr * (0.1 ** (epoch // 30))
    print('lr:',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        """


if __name__ == '__main__':
    main()

# global_step+=1
# viz.line([[test_loss, correct / len(test_loader.dataset)]],[global_step], win="test", update="append")

# python -m visdom.server -p 2333
# python train_all.py --gpu=0,1,2,3 --cfg=DMFNet_GDL_all --batch_size=4
# python train_all.py --gpu=0,1,2,3 --cfg=DNNDMF_GDL_all --batch_size=8
# python train_all.py --gpu=0,1,2 --cfg=HDC_Net --batch_size=12

