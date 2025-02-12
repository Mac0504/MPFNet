import os
import pandas as pd
import shutil
from PIL import Image

# 从原始样本中,获取起始帧至峰值帧序列
def get_onset_to_apex(code_file, img_source_path, img_dst_path):
    os.makedirs(img_source_path, exist_ok=True)
    os.makedirs(img_dst_path, exist_ok=True)
    df = pd.read_excel(code_file, engine='openpyxl')
    # 逐行读取DataFrame
    for index, row in df.iterrows():
        print(row['Subject'], '-', row['Filename'])
        for i in range(row['OnsetFrame'], row['ApexFrame'] + 1):
            img_source = img_source_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(i) + '.jpg'
            os.makedirs(img_dst_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'], exist_ok=True)
            img_dst = img_dst_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(i) + '.jpg'

            # 打开图像文件
            image = Image.open(img_source)

            # 调整图像尺寸
            resized_image = image.resize((240, 280))

            # 保存调整后的图像到目标目录
            resized_image.save(img_dst)
        print('copy success:', row['Filename'])

# 从原始样本中,获取起始帧至结束帧序列
def get_onset_to_offset(code_file, img_source_path, img_dst_path):
    os.makedirs(img_dst_path, exist_ok=True)
    df = pd.read_excel(code_file, engine='openpyxl')
    # 逐行读取DataFrame
    for index, row in df.iterrows():
        print(row['Subject'], '-', row['Filename'])
        try:
            for i in range(row['OnsetFrame'], row['OffsetFrame'] + 1):
                img_source = img_source_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(i) + '.jpg'
                os.makedirs(img_dst_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'], exist_ok=True)
                img_dst = img_dst_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(i) + '.jpg'

                # 打开图像文件
                image = Image.open(img_source)

                # 调整图像尺寸
                resized_image = image.resize((240, 280))

                # 保存调整后的图像到目标目录
                resized_image.save(img_dst)
        except Exception as e:
            print(e)

# 获取起始帧+峰值帧(放大)+结束帧 3个关键帧,其中峰值帧是放大后的
def get_onset_and_apex_amplified(code_file, img_onset_path, img_apex_path, img_offset_path, img_pair_path):
    df = pd.read_excel(code_file, engine='openpyxl')
    # 逐行读取DataFrame
    for index, row in df.iterrows():
        print(row['Subject'], '-', row['Filename'])
        img_onset = img_onset_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OnsetFrame']) + '.jpg'
        img_apex = img_apex_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['ApexFrame']) + '.jpg'
        img_offset = img_offset_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OffsetFrame']) + '.jpg'
        os.makedirs(img_pair_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'], exist_ok=True)
        img_onset_dst = img_pair_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OnsetFrame']) + '.jpg'
        img_apex_dst = img_pair_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['ApexFrame']) + '.jpg'
        img_offset_dst = img_pair_path + '\sub' + str("{:02}".format(row['Subject'])) + '\\' + row['Filename'] + '\\' + 'reg_img' + str(row['OffsetFrame']) + '.jpg'

        try:
        # 复制文件
            shutil.copy(img_onset, img_onset_dst)
            shutil.copy(img_apex, img_apex_dst)
            shutil.copy(img_offset, img_offset_dst)
        except Exception as e:
            print(e)
def main():
    code_file = r'D:\0.Malcolm\1.Micro-expression database\CASME2\CASME2-coding.xlsx'
    img_source_path = r'D:\0.Malcolm\1.Micro-expression database\CASME2\CASME2_preprocessed_Li Xiaobai\CASME2_preprocessed_Li Xiaobai\Cropped'
    img_dst_path_onset2apex = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex'
    img_dst_path_onset2offset = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2offset'

    img_onset_path = img_dst_path_onset2offset
    img_apex_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex_mag5.0'
    img_offset_path = img_dst_path_onset2offset
    img_key_frames_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_key_frames'

    # 从原始样本中,获取起始帧至峰值帧序列
    # get_onset_to_apex(code_file, img_source_path, img_dst_path_onset2apex)

    # 从原始样本中,获取起始帧至结束帧序列
    # get_onset_to_offset(code_file, img_source_path, img_dst_path_onset2offset)

    # 获取起始帧+峰值帧(放大)+结束帧 3个关键帧,其中峰值帧是放大后的
    get_onset_and_apex_amplified(code_file, img_onset_path, img_apex_path, img_offset_path, img_key_frames_path)

if __name__ == "__main__":
    main()