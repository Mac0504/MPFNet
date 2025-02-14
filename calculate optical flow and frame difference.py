import cv2
import numpy as np
import os
import shutil

def read_image(path):
    # Read an image and convert it to grayscale
    img = cv2.imread(path) 
    if img is None: 
        raise FileNotFoundError(f"Image not found at path: {path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    return img, gray_img

# Calculate optical flow
def compute_TVL1(prev, curr, bound=15):
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None) 
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(np.float32) 
    flow[flow >= 255] = 255 
    flow[flow <= 0] = 0  

    return flow

# Calculate optical strain features
def generate_optical_strain(flow):
    u = flow[...,0]
    v = flow[...,1]

    ux, uy = np.gradient(u)
    vx, vy = np.gradient(v)

    e_xy = 0.5*(uy + vx)
    e_xx = ux
    e_yy = vy
    e_m = e_xx ** 2 + 2 * e_xy ** 2 + e_yy ** 2
    e_m = np.sqrt(e_m)
    e_m = cv2.normalize(e_m, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    e_m = e_m.astype(np.uint8)

    return e_m

# Generate an optical flow image and its horizontal and vertical components
def generate_optical_flow(sample_path):
    onset = os.path.join(sample_path, 'align_onset_by_nose.png')
    apex = os.path.join(sample_path, 'apex.png')
    
    of_path = os.path.join(sample_path, 'OF.png')
    os_path = os.path.join(sample_path, 'OS.png')
    horz_path = os.path.join(sample_path, 'H.png')
    vert_path = os.path.join(sample_path, 'V.png')
    
    onset_img, onset_img_gray = read_image(onset) 
    apex_img, apex_img_gray = read_image(apex) 
    
    flow = compute_TVL1(onset_img_gray, apex_img_gray)  

    hsv = np.zeros_like(onset_img)
    hsv[..., 1] = 255 

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2 
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  

    horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')  
    vert = vert.astype('uint8')  

    optical_strain = generate_optical_strain(flow)

    cv2.imwrite(of_path, rgb)
    cv2.imwrite(os_path, optical_strain)
    cv2.imwrite(horz_path, horz)
    cv2.imwrite(vert_path, vert)
    
    print(f'generate optical flow success!: {sample_path}')


# calculate frame difference
def calculate_frame_difference(image1, image2, save_path):
    
    frame_difference = cv2.absdiff(image1, image2)
    # Convert frame difference image to grayscale image
    gray_difference = cv2.cvtColor(frame_difference, cv2.COLOR_BGR2GRAY)
    # Binarization
    _, binary_difference = cv2.threshold(gray_difference, 30, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(os.path.join(save_path, "z_difference.png"), frame_difference)
    cv2.imwrite(os.path.join(save_path, "z_difference_binary.png"), binary_difference)



exp_type = 'Micro'

source_path = os.path.join('4.pics_selected_segmented_cropped_align', exp_type)
opticalflow_path = os.path.join('5.pics_selected_segmented_cropped_align_opticalflow', exp_type)

for sample in os.listdir(source_path):
    onset_path = os.path.join(source_path, sample, 'align_onset_by_nose.png')
    apex_path = os.path.join(source_path, sample, 'apex.png')
    
    os.makedirs(os.path.join(opticalflow_path, sample), exist_ok=True)
    
    shutil.copy2(onset_path, os.path.join(opticalflow_path, sample, 'align_onset_by_nose.png'))
    shutil.copy2(apex_path, os.path.join(opticalflow_path, sample, 'apex.png'))
    
    sample_path = os.path.join(opticalflow_path, sample)
    generate_optical_flow(sample_path)