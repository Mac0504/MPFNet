import os
import concurrent.futures
import imageio
import os
import cv2
import os
import imageio
import glob
from datetime import datetime


def generate_mp4(image_folder, video_name, fps):
    frame_size = (240, 280)

    # 获取图片文件列表
    images = [img for img in os.listdir(image_folder)]
    # 按文件名排序
    images.sort()
    # print('images:', images)

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        # print('image_path:', image_path)
        img = cv2.imread(image_path)

        # 调整图片尺寸为目标尺寸
        img = cv2.resize(img, frame_size)

        # 将图片写入视频
        video.write(img)

    # 释放资源
    cv2.destroyAllWindows()
    video.release()

    print("视频已生成：", video_name)
            
            
def main():
    fps = int(100)
    single_or_mutil = 'single'

    if single_or_mutil == 'mutil':
        cropped_path = r'D:\0.Malcolm\1.Micro-expression database\CASME2\CASME2_preprocessed_Li Xiaobai\CASME2_preprocessed_Li Xiaobai\Cropped'
        video_path = r'data\examples\CASME2_video_cropped_fps' + str(fps)
        for sub in os.listdir(cropped_path):
            sub_path = os.path.join(cropped_path, sub)
            for ep in os.listdir(sub_path):
                pics_path = os.path.join(sub_path, ep)
                output_video_path = os.path.join(video_path, sub)
                os.makedirs(output_video_path, exist_ok=True)
                output_video = output_video_path + '\\' + ep + '.mp4'
                # print('output_video_path:', output_video_path)
                generate_mp4(pics_path, output_video, fps)

    if single_or_mutil == 'single':
        pics_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics_cropped_mag5.0\sub01\EP02_01f'
        output_video_name = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics_cropped_mag5.0\sub01\EP02_01f\video.mp4'
        generate_mp4(pics_path, output_video_name, fps)
            
            
if __name__ == "__main__":
    main()