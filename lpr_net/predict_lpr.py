# -*- coding: utf-8 -*-
# /usr/bin/env python3

"""
LPRNet 推理与评估脚本
Author: aiboy.wei@outlook.com + ChatGPT enhanced
"""

import os
import cv2
import torch
import argparse
import numpy as np
from torch.autograd import Variable
from models.LPRNet import CHARS, LPRNet


def parse_args():
    parser = argparse.ArgumentParser(description='LPRNet Inference')
    parser.add_argument('--img_size', default=[94, 24], help='输入图片大小 [W, H]')
    parser.add_argument('--test_img_dir', default='./traindata/train/blue', help='测试图片文件夹路径')
    parser.add_argument('--dropout_rate', type=float, default=0, help='dropout比率')
    parser.add_argument('--lpr_max_len', type=int, default=8, help='车牌最大长度')
    parser.add_argument('--weights_path', default='../weights/lpr/lpr_best.pth', help='模型权重路径')

    # 自动设置 device
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--device', default=default_device, help='使用设备')

    return parser.parse_args()


def preprocess_image(img_path, img_size):
    """读取和预处理图像"""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.resize(img, img_size)
    img = img.astype('float32')
    img = (img - 127.5) * 0.0078125
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0)  # 添加 batch 维度


def decode_prediction(preb, char_list):
    """CTC解码：去除重复和空白"""
    preb_label = [np.argmax(preb[:, j], axis=0) for j in range(preb.shape[1])]
    no_repeat = []

    pre_c = preb_label[0]
    if pre_c != len(char_list) - 1:
        no_repeat.append(char_list[pre_c])

    for c in preb_label[1:]:
        if c != pre_c and c != len(char_list) - 1:
            no_repeat.append(char_list[c])
        pre_c = c
    return ''.join(no_repeat)


def evaluate_predictions(gt, pred):
    """计算单张预测的字符级正确数"""
    min_len = min(len(gt), len(pred))
    correct_chars = sum(gt[i] == pred[i] for i in range(min_len))
    return correct_chars, len(gt)


def predict():
    args = parse_args()

    # 模型加载
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = LPRNet(lpr_max_len=args.lpr_max_len, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.to(device)
    model.eval()

    print("✅ 模型加载成功，开始推理评估...")

    # 初始化统计数据
    total_samples = 0
    correct_plates = 0
    total_chars = 0
    correct_chars = 0

    with torch.no_grad():
        for root, _, files in os.walk(args.test_img_dir):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                img_path = os.path.join(root, fname)
                gt_label = os.path.splitext(fname)[0]

                # 图像预处理
                img_tensor = preprocess_image(img_path, args.img_size).to(device)
                prebs = model(img_tensor).cpu().numpy()

                # 解码预测
                pred_label = decode_prediction(prebs[0], CHARS)

                # 输出预测信息
                print(f"标签：{gt_label} | 预测：{pred_label}")

                # 评估
                total_samples += 1
                if pred_label == gt_label:
                    correct_plates += 1

                correct, total = evaluate_predictions(gt_label, pred_label)
                correct_chars += correct
                total_chars += total

    # 汇总评估结果
    plate_acc = 100.0 * correct_plates / total_samples if total_samples else 0
    char_acc = 100.0 * correct_chars / total_chars if total_chars else 0

    print("\n📊 推理评估结果：")
    print(f"总样本数        : {total_samples}")
    print(f"完全匹配数量    : {correct_plates}")
    print(f"车牌级准确率    : {plate_acc:.2f}%")
    print(f"字符级准确率    : {char_acc:.2f}%")


if __name__ == '__main__':
    predict()
