import os

import cv2



avi_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_video_avi'
mp4_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_video_mp4'

for sub in os.listdir(avi_path):
    sub_path = os.path.join(avi_path, sub)
    # 遍历路径下的文件
    for filename in os.listdir(sub_path):
        filepath = os.path.join(sub_path, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.avi'):
            # print(filepath)
            os.makedirs(os.path.join(mp4_path, sub), exist_ok=True)
            output_video = os.path.join(mp4_path, sub, filename.split('.')[0]+'.mp4')
            # print(output_video)
            # 打开输入视频文件
            video = cv2.VideoCapture(filepath)

            # 获取输入视频的宽度、高度和帧率
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)

            # 创建视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

            # 逐帧读取输入视频并写入输出视频
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                output.write(frame)

            # 释放资源
            video.release()
            output.release()

            print("视频已转换为：", output_video)
