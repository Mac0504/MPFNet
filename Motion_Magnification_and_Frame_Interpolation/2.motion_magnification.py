import logging
import os
import time
import numpy as np
import torch
from moviepy.editor import (CompositeVideoClip, TextClip, VideoFileClip, clips_array)
from torch.functional import F
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import trange
import cv2
import torch
from torchvision import transforms
from PIL import Image

# 对单个视频进行放大
def amplify_video(model_path, input_video, output_video, *, amplification=1.0, device="cuda:0", skip_frames=1):
    device = torch.device(device)
    model = torch.load(model_path).to(device)
    video = VideoFileClip(input_video) # 使用MoviePy库的VideoFileClip函数加载输入视频，创建一个视频对象
    _to_tensor = transforms.ToTensor() # 使用PyTorch的变换函数ToTensor()创建一个将图像转换为张量的转换器
    last_frames = []
    num_skipped_frames = 5

    # 接收输入帧input_frame作为参数，用于处理每个帧
    def _video_process_frame(input_frame):
        nonlocal last_frames
        frame = _to_tensor(to_pil_image(input_frame)).to(device)
        frame = torch.unsqueeze(frame, 0)

        # 如果帧数小于num_skipped_frames，则不放大，返回原帧
        if len(last_frames) < num_skipped_frames:
            last_frames.append(frame)
            return input_frame

        # 将放大系数（amplification）转换为张量，并将其移动到指定的设备上
        amp_f_tensor = torch.tensor([[float(amplification)]], dtype=torch.float, device=device)

        # 使用模型对最近的几帧和当前帧进行处理，并获取预测帧
        pred_frame, _, _ = model.forward(last_frames[0], frame, amp_f_tensor)

        # 将预测帧从张量格式转换为图像格式，并对像素值进行裁剪和限制在0到1之间
        pred_frame = to_pil_image(pred_frame.squeeze(0).clamp(0, 1).detach().cpu())
        pred_frame = np.array(pred_frame)
        last_frames.append(frame)
        last_frames = last_frames[-num_skipped_frames:] # 保持last_frames列表中只有最近的几帧，删除旧的帧
        return pred_frame

    amp_video = video.fl_image(_video_process_frame) # 将处理函数_video_process_frame应用到每个帧上，生成增强后的视频
    amp_video.write_videofile(output_video) # 将增强后的视频保存到指定的输出路径

    print(f'amplify {output_video} success!')


# 对单个文件夹中的图片序列进行放大
def amplify_pics(model_path, pics_path, pics_mag_path, *, amplification=1.0, device="cuda:0"):
    device = torch.device(device)
    model = torch.load(model_path).to(device)
    _to_tensor = transforms.ToTensor()

    last_frames = []
    num_skipped_frames = 1
    frame_size = (240, 280)
    # frame_size = (231, 282)

    os.makedirs(pics_mag_path, exist_ok=True)
    for i, input_image in enumerate(os.listdir(pics_path)):
        input_image_path = os.path.join(pics_path, input_image)
        output_image_path = os.path.join(pics_mag_path, input_image)

        # print('input_image_path:', input_image_path) # D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics_cropped\sub04\EP19_01f\reg_img226.jpg
        # print('output_image_path:', output_image_path) # D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics_cropped_mag5.0\sub07\EP18_03\reg_img96.jpg

        input_frame = Image.open(input_image_path).resize(frame_size)
        frame = _to_tensor(input_frame).to(device)
        frame = torch.unsqueeze(frame, 0)

        # 如果帧数小于num_skipped_frames，则不放大，返回原帧
        if len(last_frames) < num_skipped_frames:
            last_frames.append(frame)

        amp_f_tensor = torch.tensor([[float(amplification)]], dtype=torch.float, device=device)

        pred_frame, _, _ = model.forward(last_frames[0], frame, amp_f_tensor)
        pred_frame = pred_frame.squeeze(0).clamp(0, 1).detach().cpu()

        pred_frame = transforms.ToPILImage()(pred_frame)
        pred_frame = np.array(pred_frame)
        gray_image = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, rgb_image)

        last_frames.append(frame)
        last_frames = last_frames[-num_skipped_frames:]

    print(f'amplify {pics_path} success!')


# 通过循环批量对视频进行放大
def mag_videos(model_path, video_path, video_mag_path, amplification):
    for sub in os.listdir(video_path):
        sub_path = os.path.join(video_path, sub)
        # 遍历路径下的文件
        for filename in os.listdir(sub_path):
            filepath = os.path.join(sub_path, filename)
            if os.path.isfile(filepath) and filename.lower().endswith('.mp4'):
                # print(filepath)
                os.makedirs(os.path.join(video_mag_path, sub), exist_ok=True)
                video_mag = os.path.join(video_mag_path, sub, filename.split('.')[0] + '_mag.mp4')
                # print(output_video)
                # 打开输入视频文件
                # video = cv2.VideoCapture(filepath)
                amplify_video(model_path=model_path, input_video=filepath, output_video=video_mag, amplification=amplification)


# 对多个文件夹中的图片序列进行放大
def mag_pics(model_path, pics_path, pics_mag_path, amplification):
    os.makedirs(pics_mag_path, exist_ok=True )
    for sub in os.listdir(pics_path):
        print('sub:', sub)
        sub_path = os.path.join(pics_path, sub)
        # print('sub_path:', sub_path)
        for ep in os.listdir(sub_path):
            pics_folder = os.path.join(sub_path, ep)
            output_pics_path = os.path.join(pics_mag_path, sub, ep)
            os.makedirs(output_pics_path, exist_ok=True)
            # print('pics_folder:', pics_folder) # D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics_cropped\sub26\EP18_51
            # print('output_pics_path:', output_pics_path) # D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics_cropped_mag5.0\sub26\EP18_51
            amplify_pics(model_path=model_path, pics_path=pics_folder, pics_mag_path=output_pics_path, amplification=amplification)

def main():
    amplification = 2
    model_path = 'data/models_motion_magnification/20191204-b4-r0.1-lr0.0001-05.pt'

    # 对单个视频进行放大
    # amplify_video(model_path=model_path, input_video=r'data/examples/baby.mp4', output_video=r'data/examples/baby_mag.mp4', amplification=amplification)

    # 对单个文件夹中的图片序列进行放大
    pics_folder = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex\sub01\EP19_05f'
    output_pics_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex_mag' + str(amplification) + '\sub01\EP19_05f'
    # amplify_pics(model_path=model_path, pics_path=pics_folder, pics_mag_path=output_pics_path, amplification=amplification)

    # 对单个文件夹中的图片序列进行放大（放大系数2~20）
    for i in range(2, 21):
        print(f'amplification is: {i}')
        pics_folder = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex\sub14\EP09_06'
        output_pics_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex_mag' + str(i) + '\sub14\EP09_06'
        amplify_pics(model_path=model_path, pics_path=pics_folder, pics_mag_path=output_pics_path, amplification=i)

    # 对多个文件夹中的图片序列进行放大
    pics_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex'
    pics_mag_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex_mag' + str(amplification)
    # mag_pics(model_path, pics_path, pics_mag_path, amplification)

    # video_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_video\CASME2_video_cropped_fps100_raw'
    # video_mag_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_video\CASME2_video_cropped_mag' + str(amplification)

    pics_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics_cropped'
    pics_mag_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics_cropped_mag' + str(amplification)

    # mag_videos(model_path, video_path, video_mag_path, amplification)
    # mag_pics(model_path, pics_path, pics_mag_path, amplification)



if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    #
    # input_video = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_video_mp4\sub01\EP02_01f.mp4'
    # amplify(model_path=model_path, input_video=input_video, amplification=3.0)
    main()