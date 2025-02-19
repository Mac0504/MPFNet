import os
import pandas as pd
import shutil
from PIL import Image

# Get the sequence of frames from onset to apex from the original samples
def get_onset_to_apex(code_file, img_source_path, img_dst_path):
    os.makedirs(img_source_path, exist_ok=True)
    os.makedirs(img_dst_path, exist_ok=True)
    df = pd.read_excel(code_file, engine='openpyxl')
    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        print(row['Subject'], '-', row['Filename'])
        # Loop through frames from onset to apex
        for i in range(row['OnsetFrame'], row['ApexFrame'] + 1):
            # Construct the source image path
            img_source = img_source_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(i) + '.jpg'
            os.makedirs(img_dst_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'], exist_ok=True)
            # Construct the destination image path
            img_dst = img_dst_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(i) + '.jpg'

            # Open the image file
            image = Image.open(img_source)

            # Resize the image
            resized_image = image.resize((240, 280))

            # Save the resized image to the destination path
            resized_image.save(img_dst)
        print('Copy success:', row['Filename'])

# Get the sequence of frames from onset to offset from the original samples
def get_onset_to_offset(code_file, img_source_path, img_dst_path):
    os.makedirs(img_dst_path, exist_ok=True)
    df = pd.read_excel(code_file, engine='openpyxl')
    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        print(row['Subject'], '-', row['Filename'])
        try:
            # Loop through frames from onset to offset
            for i in range(row['OnsetFrame'], row['OffsetFrame'] + 1):
                # Construct the source image path
                img_source = img_source_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(i) + '.jpg'
                os.makedirs(img_dst_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'], exist_ok=True)
                # Construct the destination image path
                img_dst = img_dst_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(i) + '.jpg'

                # Open the image file
                image = Image.open(img_source)

                # Resize the image
                resized_image = image.resize((240, 280))

                # Save the resized image to the destination path
                resized_image.save(img_dst)
        except Exception as e:
            print(e)

# Get the three key frames: onset frame, amplified apex frame, and offset frame
# The apex frame is amplified
def get_onset_and_apex_amplified(code_file, img_onset_path, img_apex_path, img_offset_path, img_pair_path):
    df = pd.read_excel(code_file, engine='openpyxl')
    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        print(row['Subject'], '-', row['Filename'])
        
        # Construct paths for the onset, apex, and offset frames
        img_onset = img_onset_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OnsetFrame']) + '.jpg'
        img_apex = img_apex_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['ApexFrame']) + '.jpg'
        img_offset = img_offset_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OffsetFrame']) + '.jpg'

        os.makedirs(img_pair_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'], exist_ok=True)

        # Construct destination paths for each key frame
        img_onset_dst = img_pair_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OnsetFrame']) + '.jpg'
        img_apex_dst = img_pair_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['ApexFrame']) + '.jpg'
        img_offset_dst = img_pair_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OffsetFrame']) + '.jpg'

        try:
            # Copy the onset, amplified apex, and offset frames to the destination folder
            shutil.copy(img_onset, img_onset_dst)
            shutil.copy(img_apex, img_apex_dst)
            shutil.copy(img_offset, img_offset_dst)
        except Exception as e:
            print(e)

# Main function to define paths and execute the frame extraction functions
def main():
    code_file = r'D:\0.Malcolm\1.Micro-expression database\CASME2\CASME2-coding.xlsx'
    img_source_path = r'D:\0.Malcolm\1.Micro-expression database\CASME2\CASME2_preprocessed_Li Xiaobai\CASME2_preprocessed_Li Xiaobai\Cropped'
    img_dst_path_onset2apex = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex'
    img_dst_path_onset2offset = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2offset'

    img_onset_path = img_dst_path_onset2offset
    img_apex_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex_mag5.0'
    img_offset_path = img_dst_path_onset2offset
    img_key_frames_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_key_frames'

    # Uncomment to extract frames from onset to apex
    # get_onset_to_apex(code_file, img_source_path, img_dst_path_onset2apex)

    # Uncomment to extract frames from onset to offset
    # get_onset_to_offset(code_file, img_source_path, img_dst_path_onset2offset)

    # Extract three key frames: onset, amplified apex, and offset
    get_onset_and_apex_amplified(code_file, img_onset_path, img_apex_path, img_offset_path, img_key_frames_path)

if __name__ == "__main__":
    main()
