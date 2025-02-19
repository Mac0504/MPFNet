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
from PIL import Image

# Function to amplify a single video
def amplify_video(model_path, input_video, output_video, *, amplification=1.0, device="cuda:0", skip_frames=1):
    device = torch.device(device)
    model = torch.load(model_path).to(device)  # Load the model onto the specified device
    video = VideoFileClip(input_video)  # Load the input video using MoviePy
    _to_tensor = transforms.ToTensor()  # Define a transformation to convert images to tensors
    last_frames = []  # List to store previous frames
    num_skipped_frames = 5  # Number of frames to skip in processing

    # Frame processing function that gets applied to each frame in the video
    def _video_process_frame(input_frame):
        nonlocal last_frames
        frame = _to_tensor(to_pil_image(input_frame)).to(device)  # Convert the frame to a tensor
        frame = torch.unsqueeze(frame, 0)  # Add batch dimension

        # If the number of frames processed is less than the skipped frames, return the original frame
        if len(last_frames) < num_skipped_frames:
            last_frames.append(frame)
            return input_frame

        # Convert the amplification factor to a tensor and move it to the device
        amp_f_tensor = torch.tensor([[float(amplification)]], dtype=torch.float, device=device)

        # Use the model to process the last few frames and the current frame to predict the amplified frame
        pred_frame, _, _ = model.forward(last_frames[0], frame, amp_f_tensor)

        # Convert the predicted frame back to an image, clamp the pixel values between 0 and 1, and detach from the computation graph
        pred_frame = to_pil_image(pred_frame.squeeze(0).clamp(0, 1).detach().cpu())
        pred_frame = np.array(pred_frame)
        last_frames.append(frame)
        last_frames = last_frames[-num_skipped_frames:]  # Keep only the last 'num_skipped_frames' frames
        return pred_frame

    # Apply the frame processing function to each frame in the video and create the amplified video
    amp_video = video.fl_image(_video_process_frame)
    amp_video.write_videofile(output_video)  # Save the amplified video

    print(f'Amplification of {output_video} was successful!')

# Function to amplify images in a folder (image sequence)
def amplify_pics(model_path, pics_path, pics_mag_path, *, amplification=1.0, device="cuda:0"):
    device = torch.device(device)
    model = torch.load(model_path).to(device)
    _to_tensor = transforms.ToTensor()

    last_frames = []
    num_skipped_frames = 1
    frame_size = (240, 280)  # Frame size for resizing images

    os.makedirs(pics_mag_path, exist_ok=True)
    for i, input_image in enumerate(os.listdir(pics_path)):
        input_image_path = os.path.join(pics_path, input_image)
        output_image_path = os.path.join(pics_mag_path, input_image)

        input_frame = Image.open(input_image_path).resize(frame_size)  # Resize the input frame
        frame = _to_tensor(input_frame).to(device)  # Convert the image to a tensor
        frame = torch.unsqueeze(frame, 0)  # Add batch dimension

        # If fewer frames have been processed than num_skipped_frames, don't amplify, return original frame
        if len(last_frames) < num_skipped_frames:
            last_frames.append(frame)

        amp_f_tensor = torch.tensor([[float(amplification)]], dtype=torch.float, device=device)

        # Process the current and last frames using the model
        pred_frame, _, _ = model.forward(last_frames[0], frame, amp_f_tensor)
        pred_frame = pred_frame.squeeze(0).clamp(0, 1).detach().cpu()  # Clamp and detach the prediction

        # Convert the predicted frame to an image and save it
        pred_frame = transforms.ToPILImage()(pred_frame)
        pred_frame = np.array(pred_frame)
        gray_image = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale (optional)
        rgb_image = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        cv2.imwrite(output_image_path, rgb_image)  # Save the image

        last_frames.append(frame)
        last_frames = last_frames[-num_skipped_frames:]  # Keep only the most recent frame

    print(f'Amplification of {pics_path} was successful!')

# Function to amplify videos in a folder (batch processing)
def mag_videos(model_path, video_path, video_mag_path, amplification):
    for sub in os.listdir(video_path):
        sub_path = os.path.join(video_path, sub)
        # Loop through the video files in each folder
        for filename in os.listdir(sub_path):
            filepath = os.path.join(sub_path, filename)
            if os.path.isfile(filepath) and filename.lower().endswith('.mp4'):
                os.makedirs(os.path.join(video_mag_path, sub), exist_ok=True)
                video_mag = os.path.join(video_mag_path, sub, filename.split('.')[0] + '_mag.mp4')
                amplify_video(model_path=model_path, input_video=filepath, output_video=video_mag, amplification=amplification)

# Function to amplify images in multiple folders
def mag_pics(model_path, pics_path, pics_mag_path, amplification):
    os.makedirs(pics_mag_path, exist_ok=True)
    for sub in os.listdir(pics_path):
        print('Processing subfolder:', sub)
        sub_path = os.path.join(pics_path, sub)
        for ep in os.listdir(sub_path):
            pics_folder = os.path.join(sub_path, ep)
            output_pics_path = os.path.join(pics_mag_path, sub, ep)
            os.makedirs(output_pics_path, exist_ok=True)
            amplify_pics(model_path=model_path, pics_path=pics_folder, pics_mag_path=output_pics_path, amplification=amplification)

def main():
    amplification = 2
    model_path = 'data/models_motion_magnification/20191204-b4-r0.1-lr0.0001-05.pt'  # Model path

    # Example of amplifying a single video (commented out)
    # amplify_video(model_path=model_path, input_video=r'data/examples/baby.mp4', output_video=r'data/examples/baby_mag.mp4', amplification=amplification)

    # Example of amplifying a single folder of images (commented out)
    pics_folder = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex\sub01\EP19_05f'
    output_pics_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex_mag' + str(amplification) + '\sub01\EP19_05f'
    # amplify_pics(model_path=model_path, pics_path=pics_folder, pics_mag_path=output_pics_path, amplification=amplification)

    # Batch process image amplification for multiple folders with amplification factors from 2 to 20
    for i in range(2, 21):
        print(f'Amplification factor is: {i}')
        pics_folder = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex\sub14\EP09_06'
        output_pics_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex_mag' + str(i) + '\sub14\EP09_06'
        amplify_pics(model_path=model_path, pics_path=pics_folder, pics_mag_path=output_pics_path, amplification=i)

    # Uncomment below for batch processing of multiple image folders (optional)
    # pics_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex'
    # pics_mag_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex_mag' + str(amplification)
    # mag_pics(model_path, pics_path, pics_mag_path, amplification)

    # Uncomment below for batch processing of video amplification (optional)
    # video_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_video\CASME2_video_cropped_fps100_raw'
    # video_mag_path = r'D:\0.Malcolm\2.Projects\3.motion-magnification-master\data\examples\CASME2_video\CASME2_video_cropped_mag' + str(amplification)
    # mag_videos(model_path, video_path, video_mag_path, amplification)

if __name__ == "__main__":
    main()
