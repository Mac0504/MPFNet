import re
import time
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from alibabacloud_facebody20191230.client import Client
from alibabacloud_facebody20191230.models import DetectFaceAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions
from PIL import Image


# Calling Alibaba Cloud for facial detection
def face_reg_alibabacloud(pic_path):
    config = Config(
    # To create an AccessKey ID and AccessKey Secret, please refer to https://help.aliyun.com/document_detail/175144.html
    # Read the configured AccessKey ID and AccessKey Secret from the environment variables. You must configure the environment variables before running the code example.
    access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
    access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    # Visited domain name
    endpoint='facebody.cn-shanghai.aliyuncs.com',
    # The region corresponding to the visited domain name
    region_id='cn-shanghai')

    detect_face_request = DetectFaceAdvanceRequest()

    stream = open(pic_path, 'rb')
    detect_face_request.image_urlobject = stream
    detect_face_request.landmark = True
    detect_face_request.quality = True
    detect_face_request.pose = False
    detect_face_request.max_face_number = 1

    runtime = RuntimeOptions()

    # Initialize Client
    client = Client(config)
    response = client.detect_face_advance(detect_face_request, runtime)
    # Get recognition results
    result = dict()
    result['body'] = response.body.to_map()
    face_info = result['body'].get('Data')
    # print(face_info)
    # print(face_info.get('FaceRectangles'))
    # Returns the face rectangle, which is [left, top, width, height]
    FaceRectangles = face_info.get('FaceRectangles')
    # Returns the facial feature point positioning result. For each face, a set of feature point positions are returned, expressed as (x0, y0, x1, y1, ...). 
    # If there are multiple faces, they are returned in sequence with the positioning floating point number returned.
    # LandmarkCount The number of facial landmarks is currently fixed at 105 points. They are: 24 points for eyebrows, 32 points for eyes, 6 points for nose, 34 points for mouth, and 9 points for outer contours.
    Landmarks = face_info.get('Landmarks')

    time.sleep(1)

    return FaceRectangles, Landmarks


def draw_rectangle_and_landmarks(img_path, img_save_patn):
    image = cv2.imread(img_path)
    # Calculate the bounding box of the face area
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    F, L = face_reg_alibabacloud(img_path)
    # Convert the representation to a list of coordinate pairs
    landmarks = [(int(L[i]), int(L[i + 1])) for i in range(0, len(L), 2)]
    
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # Draw facial keypoint index
    for idx, landmark in enumerate(landmarks):
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        
    cv2.imwrite(img_save_patn, image)
    print('save img success!')


# Crop the face area
def get_face_roi(src_img, img_save_path):
    FaceRectangles, Landmarks = face_reg_alibabacloud(os.path.join(me_crop_path, me_sample, src_img))
    image = cv2.imread(os.path.join(me_crop_path, me_sample, src_img))

    face_roi = image[FaceRectangles[1]: FaceRectangles[1]+FaceRectangles[3],
                     FaceRectangles[0]: FaceRectangles[0]+FaceRectangles[2]]


    cv2.imwrite(img_save_path, face_roi)
    print('save img success!')


# Calculate the nose center of mass
def calculate_nose_centroid(landmarks):
    points_nose = np.array([[landmarks[i][0], landmarks[i][1]] for i in range(58, 63)])
    return points_nose.mean(axis=0)


# Apply an affine transformation to align the images' centroids
def align_images_by_nose_centroid(src_image, src_landmarks, dst_image, dst_landmarks):
    src_nose_centroid = calculate_nose_centroid(src_landmarks)
    dst_nose_centroid = calculate_nose_centroid(dst_landmarks)

    # Calculate the translation vector
    tx = dst_nose_centroid[0] - src_nose_centroid[0]
    ty = dst_nose_centroid[1] - src_nose_centroid[1]

    # Creating a translation matrix
    M_translation = np.float32([[1, 0, tx], [0, 1, ty]])

    # Get the size of the target image
    h, w = dst_image.shape[:2]

    # Apply a translation transform
    aligned_image = cv2.warpAffine(src_image, M_translation, (w, h))

    return aligned_image


# Face Alignment
def get_align_image(onset_path, apex_path, aligned_onset_path):
    onset = cv2.imread(onset_path)
    apex = cv2.imread(apex_path)
    
    F_onset, L_onset = face_reg_alibabacloud(onset_path)
    # Convert the representation to a list of coordinate pairs
    landmarks_onset = [(int(L_onset[i]), int(L_onset[i + 1])) for i in range(0, len(L_onset), 2)]
    F_apex, L_apex = face_reg_alibabacloud(apex_path)
    landmarks_apex = [(int(L_apex[i]), int(L_apex[i + 1])) for i in range(0, len(L_apex), 2)]

    # Center of mass alignment
    onset_aligned = align_images_by_nose_centroid(onset, landmarks_onset, apex, landmarks_apex)

    # Save the centroid-aligned image to a file
    cv2.imwrite(aligned_onset_path, onset_aligned)

    print(f'{aligned_onset_path}')
    print('align_image success！')
