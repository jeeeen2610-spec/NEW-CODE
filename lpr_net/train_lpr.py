# -*- coding: utf-8 -*-
# /usr/bin/env/python3


from torch.utils.data import DataLoader

from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os

from tqdm import tqdm

from models.LPRNet import LPRNet, CHARS
from datasets.dataloader import LPRDataset


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=1, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs',
                        default=r"F:\lg\BaiduSyncdisk\project\person_code\project_self\1_code\1_chepai_OCR\data\traindata\LPR_traindata\LPRnet_data\train",
                        help='the train images path')
    parser.add_argument('--test_img_dirs_blue', default=r"F:\lg\BaiduSyncdisk\project\person_code\project_self\1_code\1_chepai_OCR\data\traindata\LPR_traindata\LPRnet_data\val\blue",
                        help='the test images path')
    parser.add_argument('--test_img_dirs_green', default=r"F:\lg\BaiduSyncdisk\project\person_code\project_self\1_code\1_chepai_OCR\data\traindata\LPR_traindata\LPRnet_data\val\green",
                        help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.001, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=64, help='training batch size.')
    parser.add_argument('--test_batch_size', default=64, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=20, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=20, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[20, 40, 60, 80, 100], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default=r'./runs/', help='Location to save checkpoint models')
    parser.add_argument('--pretrained_model', default=r'F:\lg\BaiduSyncdisk\project\person_code\project_self\1_code\1_chepai_OCR\code24_yolov8\weights\lpr\lpr_best.pth', help='no pretrain')

    args = parser.parse_args()

    return args


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int32)
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


def train():
    args = get_parser()

    T_length = 18

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    lprnet = LPRNet(lpr_max_len=args.lpr_max_len,
                    class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # 加载预训练模型
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    else:
        lprnet.weights_init(lprnet)
        print("initial net weights successful!")

    # define optimizer
    optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha=0.9, eps=1e-08,
    #                           momentum=args.momentum, weight_decay=args.weight_decay)

    train_dataset = LPRDataset(args.train_img_dirs, args.img_size, args.lpr_max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn)

    test_dataset = LPRDataset(args.test_img_dirs_blue, args.img_size, args.lpr_max_len)
    test_dataloader_blue = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=collate_fn)

    test_dataset = LPRDataset(args.test_img_dirs_green, args.img_size, args.lpr_max_len)
    test_dataloader_green = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                      num_workers=args.num_workers, collate_fn=collate_fn)

    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')

    best_loss = 10000
    for epoch in range(0, args.max_epoch):
        lprnet.train()
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar, total=len(train_dataloader))
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        for i, data in pbar:
            images, labels, lengths = data
            # get ctc parameters
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)

            if args.cuda:
                images = Variable(images, requires_grad=False).cuda()
                labels = Variable(labels, requires_grad=False).cuda()
            else:
                images = Variable(images, requires_grad=False)
                labels = Variable(labels, requires_grad=False)

            # forward
            logits = lprnet(images)
            log_probs = logits.permute(2, 0, 1)  # for ctc loss: T x N x C
            log_probs = log_probs.log_softmax(2).requires_grad_()
            # log_probs = log_probs.detach().requires_grad_()
            # backprop
            optimizer.zero_grad()

            # log_probs: 预测结果 [18, bs, 68]  其中18为序列长度  68为字典数
            # labels: [93]
            # input_lengths:  tuple   example: 000=18  001=18...   每个序列长度
            # target_lengths: tuple   example: 000=7   001=8 ...   每个gt长度
            loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            # if loss.item() == np.inf:
            #     continue
            loss.backward()
            optimizer.step()

            pbar.set_description(('%2s train_loss:%2.7s lr:%2.6s') %
                                 (f'{epoch}/{args.max_epoch - 1}', loss.item(), lr))

            # 保存训练损失最低的模型, epoch > 3 时
            if loss.item() < best_loss and epoch > 3:
                model_name = os.path.join(args.save_folder, "best.pth")
                torch.save(lprnet.state_dict(), model_name)
                best_loss = loss.item()

        if epoch != 0 and epoch % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(epoch) + '.pth')

        if epoch % args.test_interval == 0:
            print('蓝牌测试准确度：')
            Greedy_Decode_Eval(lprnet, test_dataloader_blue, args)
            print('绿牌测试准确度：')
            Greedy_Decode_Eval(lprnet, test_dataloader_green, args)

    print("Final test Accuracy:")
    print('蓝牌测试准确度：Tp表示正确预测的样本数，Tn_1表示预测标签长度与真实标签长度不一致的样本数，Tn_2表示预测标签与真实标签不完全匹配的样本数。')
    Greedy_Decode_Eval(lprnet, test_dataloader_blue, args)
    print('绿牌测试准确度：Tp表示正确预测的样本数，Tn_1表示预测标签长度与真实标签长度不一致的样本数，Tn_2表示预测标签与真实标签不完全匹配的样本数。')
    Greedy_Decode_Eval(lprnet, test_dataloader_green, args)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'last.pth')


def Greedy_Decode_Eval(net, test_dataloader, args):
    # Tp、Tn_1、Tn_2：这些变量用于统计正确预测的样本数。
    # Tp表示正确预测的样本数，Tn_1表示预测标签长度与真实标签长度不一致的样本数，Tn_2表示预测标签与真实标签不完全匹配的样本数。
    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    pbar = enumerate(test_dataloader)
    pbar = tqdm(pbar, total=len(test_dataloader))
    for i, data in pbar:
        images, labels, lengths = data

        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start + length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = []
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = []
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} 'Tp':{}, 'Tn_1':{}, 'Tn_2':{}, 'all_num':{}".format(Acc, Tp, Tn_1, Tn_2, (Tp + Tn_1 + Tn_2)))



if __name__ == "__main__":
    train()
