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

sys.path.append('.')  # Add current directory to system path for importing local modules
warnings.filterwarnings('ignore')

def interpolation(args, pic1, pic2, pics_save_path):
    '''==========Model setting=========='''
    # Set different models and related configurations based on the command-line argument args.model
    TTA = True
    if args.model == 'ours_small_t':
        TTA = False
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=16,  # Feature channels
            depth=[2, 2, 2, 2, 2]  # Network depth configuration
        )
    else:
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=32,  # Feature channels
            depth=[2, 2, 2, 4, 4]  # Network depth configuration
        )

    model = Model(-1)
    model.load_model()  # Load model weights
    model.eval()  # Set model to evaluation mode
    model.device()  # Set device (GPU/CPU)

    print(f'=========================Start Generating=========================')

    I0 = cv2.imread(pic1)  # Read the first image (onset frame)
    I2 = cv2.imread(pic2)  # Read the second image (apex frame)

    # Convert I0 to PyTorch tensor, perform preprocessing: transpose, normalize, and add batch dimension
    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)  # torch.Size([1, 3, 480, 640])
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

    # Create an instance of InputPadder to pad the image to a multiple of 32 (for efficient processing)
    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)  # Apply padding

    # Add the original I0 image (in RGB format) to the list of images
    images = [I0[:, :, ::-1]]  # Reverse the channel order from BGR (OpenCV default) to RGB

    # Perform multiple inference operations on the padded images using the model
    preds = model.multi_inference(
        I0_, I2_, TTA=TTA,
        time_list=[(i + 1) * (1. / args.n) for i in range(args.n - 1)],  # Interpolation time list
        fast_TTA=TTA
    )

    # For each predicted frame, unpad and adjust the image dimensions, and append it to the images list
    for pred in preds:
        images.append(
            (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1]
        )

    # Add the original I2 image (in RGB format) to the images list
    images.append(I2[:, :, ::-1])
    images = np.asarray(images)

    print("len(images):", len(images))  # Should be 17 (1 initial, n-1 interpolated, 1 final)

    # Save each frame as a separate image file
    for i, frame_data in enumerate(images):
        image = Image.fromarray(frame_data)  # Convert NumPy array to an Image object
        image.save(pics_save_path + f'/frame_{i}.png')  # Save the frame

    # Create and save the GIF using the generated frames
    mimsave(pics_save_path + f'/gif_out_{str(args.model)}_{str(args.n)}x40.gif', images, 'GIF', duration=int(1 / 40 * 1000))

    print(f'=========================Done=========================')

def main():
    # Create a command-line argument parser
    parser = argparse.ArgumentParser()

    # Add --model argument to specify the model to use, default is 'ours_t'
    parser.add_argument('--model', default='ours_t', type=str)
    parser.add_argument('--n', default=11, type=int)  # Specify the number of frames to generate, default is 11
    args = parser.parse_args()  # Parse the command-line arguments

    # Ensure the specified model exists
    assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'

    code_file = r'D:\0.Malcolm\1.Micro-expression database\CASME2\CASME2-coding.xlsx'
    img_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_key_frames'
    df = pd.read_excel(code_file, engine='openpyxl')  # Read the Excel file containing the metadata

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        print(row['Subject'], '-', row['Filename'])
        # Construct the file paths for the onset and apex frames
        pic1 = img_path + '\\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OnsetFrame']) + '.jpg'
        pic2 = img_path + '\\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['ApexFrame']) + '.jpg'
        
        # Define the path to save the generated frames
        interpolation_frames_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_interpolation_frames'
        pics_save_path = os.path.join(interpolation_frames_path, 'sub' + str("{:02}".format(row['Subject'])), row['Filename'])
        print('pics_save_path:', pics_save_path)
        
        os.makedirs(pics_save_path, exist_ok=True)  # Create the directory if it does not exist

        # Call the interpolation function to generate frames and save them
        interpolation(args, pic1, pic2, pics_save_path)

if __name__ == '__main__':
    main()
