import cv2
import math
import sys
import torch
import pandas as pd
from PIL import Image
import numpy as np
import argparse
from imageio import mimsave
import warnings
import os
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
sys.path.append('.') # 将当前目录添加到系统路径中，以便导入本地模块
warnings.filterwarnings('ignore')


def interpolation(args, pic1, pic2, pics_save_path):
    '''==========Model setting=========='''
    # 根据命令行参数 args.model 的值，设置不同的模型和相关配置
    TTA = True
    if args.model == 'ours_small_t':
        TTA = False
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=16,
            depth=[2, 2, 2, 2, 2]
        )
    else:
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=32,
            depth=[2, 2, 2, 4, 4]
        )
    model = Model(-1)
    model.load_model()  # 加载模型的权重
    model.eval()  # 将模型设置为评估模式
    model.device()

    print(f'=========================Start Generating=========================')

    I0 = cv2.imread(pic1)
    I2 = cv2.imread(pic2)

    # 将 I0 转换为 PyTorch 张量，并进行一些预处理操作，如转置维度、归一化和添加批次维度
    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(
        0)  # torch.Size([1, 3, 480, 640])  .unsqueeze(0)：在第 0 维上添加一个额外的维度，用于表示批次大小
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

    # 创建一个 InputPadder 类的实例，传入 I0_.shape（图像形状）和 divisor=32（除数）作为参数，进行填充操作
    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)  # torch.Size([1, 3, 480, 640])

    # 将原始的 I0 图像添加到 images 列表中
    images = [I0[:, :, ::-1]]  # ::-1 表示反转通道维度，将 BGR 顺序（OpenCV 默认的通道顺序）转换为 RGB 顺序

    # 使用 model 对填充后的图像进行多次推理操作，并将结果存储在 preds 中，15*torch.Size([3, 480, 640])
    preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i + 1) * (1. / args.n) for i in range(args.n - 1)], fast_TTA=TTA)

    # 对于 preds 中的每个预测结果
    for pred in preds:
        # 裁剪、转换为 NumPy 数组、调整维度顺序、还原像素值范围并对图像进行反转，然后将图像添加到 images 列表中
        images.append((padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])

    # 将原始的 I2 图像添加到 images 列表中
    images.append(I2[:, :, ::-1])
    images = np.asarray(images)

    print("len(images):", len(images))  # 17

    # 逐帧保存为图片
    for i, frame_data in enumerate(images):
        # 创建Image对象
        image = Image.fromarray(frame_data)

        # 保存图片
        image.save(pics_save_path + f'/frame_{i}.png')

    # 保存为gif
    mimsave(pics_save_path + f'/gif_out_{str(args.model)}_{str(args.n)}x40.gif', images, 'GIF', duration=int(1 / 40 * 1000))

    print(f'=========================Done=========================')

def main():
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()
    # 向参数解析器添加一个 --model 参数，指定要使用的模型，默认为 'ours_t'
    parser.add_argument('--model', default='ours_t', type=str)
    parser.add_argument('--n', default=15, type=int)  # 指定生成图像时使用的帧数，默认为 16
    args = parser.parse_args()  # 解析命令行参数，并将结果存储在 args 变量中
    assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'

    code_file = r'D:\0.Malcolm\1.Micro-expression database\CASME2\CASME2-coding.xlsx'
    img_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_key_frames'
    df = pd.read_excel(code_file, engine='openpyxl')

    # 逐行读取DataFrame
    for index, row in df.iterrows():
        print(row['Subject'], '-', row['Filename'])
        pic1 = img_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OnsetFrame']) + '.jpg'
        pic2 = img_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['ApexFrame']) + '.jpg'
        interpolation_frames_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_interpolation_frames'
        pics_save_path = os.path.join(interpolation_frames_path, 'sub' + str("{:02}".format(row['Subject'])), row['Filename'])
        print('pics_save_path:', pics_save_path)
        os.makedirs(pics_save_path, exist_ok=True)

        interpolation(args, pic1, pic2, pics_save_path)

if __name__ == '__main__':
    main()
