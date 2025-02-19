import cv2
import numpy as np
import os
import shutil

def read_image(path):
    # Read an image from the specified path and convert it to grayscale
    img = cv2.imread(path) 
    if img is None: 
        raise FileNotFoundError(f"Image not found at path: {path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    return img, gray_img

# Calculate optical flow using the TV-L1 method
def compute_TVL1(prev, curr, bound=15):
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()  # Create optical flow object
    flow = TVL1.calc(prev, curr, None)  # Compute optical flow
    assert flow.dtype == np.float32  # Ensure the flow data type is float32

    # Normalize the flow values to a range of [0, 255]
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(np.float32) 
    flow[flow >= 255] = 255  # Cap the maximum flow value at 255
    flow[flow <= 0] = 0  # Ensure the flow values are not less than 0

    return flow

# Generate optical strain features from the optical flow field
def generate_optical_strain(flow):
    # Extract horizontal and vertical components of the flow
    u = flow[...,0]
    v = flow[...,1]

    # Compute the gradients of the flow components
    ux, uy = np.gradient(u)
    vx, vy = np.gradient(v)

    # Calculate the optical strain tensors
    e_xy = 0.5 * (uy + vx)
    e_xx = ux
    e_yy = vy
    e_m = e_xx ** 2 + 2 * e_xy ** 2 + e_yy ** 2  # Compute strain magnitude
    e_m = np.sqrt(e_m)
    
    # Normalize the strain magnitude to the range [0, 255]
    e_m = cv2.normalize(e_m, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    e_m = e_m.astype(np.uint8)

    return e_m

# Generate optical flow image and its horizontal and vertical components
def generate_optical_flow(sample_path):
    # Define paths for onset and apex images
    onset = os.path.join(sample_path, 'align_onset_by_nose.png')
    apex = os.path.join(sample_path, 'apex.png')
    
    # Define output file paths for optical flow images and components
    of_path = os.path.join(sample_path, 'OF.png')
    os_path = os.path.join(sample_path, 'OS.png')
    horz_path = os.path.join(sample_path, 'H.png')
    vert_path = os.path.join(sample_path, 'V.png')
    
    # Read onset and apex images
    onset_img, onset_img_gray = read_image(onset) 
    apex_img, apex_img_gray = read_image(apex) 
    
    # Compute the optical flow between the onset and apex images
    flow = compute_TVL1(onset_img_gray, apex_img_gray)  

    # Convert optical flow to HSV color representation for visualization
    hsv = np.zeros_like(onset_img)
    hsv[..., 1] = 255  # Set saturation to maximum

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # Compute magnitude and angle of the flow
    hsv[..., 0] = ang * 180 / np.pi / 2  # Map angle to hue
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Normalize magnitude to [0, 255]
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Convert HSV to BGR for visualization

    # Normalize horizontal and vertical flow components to [0, 255]
    horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')  # Convert to unsigned 8-bit integer type
    vert = vert.astype('uint8')  # Convert to unsigned 8-bit integer type

    # Generate optical strain map from the flow
    optical_strain = generate_optical_strain(flow)

    # Save the generated images
    cv2.imwrite(of_path, rgb)
    cv2.imwrite(os_path, optical_strain)
    cv2.imwrite(horz_path, horz)
    cv2.imwrite(vert_path, vert)
    
    print(f'Generate optical flow success!: {sample_path}')


# Calculate frame difference between two images
def calculate_frame_difference(image1, image2, save_path):
    # Compute absolute difference between the two frames
    frame_difference = cv2.absdiff(image1, image2)
    
    # Convert frame difference image to grayscale
    gray_difference = cv2.cvtColor(frame_difference, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding to highlight significant differences
    _, binary_difference = cv2.threshold(gray_difference, 30, 255, cv2.THRESH_BINARY)
    
    # Save the frame difference and binary difference images
    cv2.imwrite(os.path.join(save_path, "z_difference.png"), frame_difference)
    cv2.imwrite(os.path.join(save_path, "z_difference_binary.png"), binary_difference)


# Define experiment type and file paths
exp_type = 'Micro'

source_path = os.path.join('4.pics_selected_segmented_cropped_align', exp_type)
opticalflow_path = os.path.join('5.pics_selected_segmented_cropped_align_opticalflow', exp_type)

# Process each sample in the source directory
for sample in os.listdir(source_path):
    onset_path = os.path.join(source_path, sample, 'align_onset_by_nose.png')
    apex_path = os.path.join(source_path, sample, 'apex.png')
    
    # Create output directory for each sample and copy the necessary images
    os.makedirs(os.path.join(opticalflow_path, sample), exist_ok=True)
    shutil.copy2(onset_path, os.path.join(opticalflow_path, sample, 'align_onset_by_nose.png'))
    shutil.copy2(apex_path, os.path.join(opticalflow_path, sample, 'apex.png'))
    
    # Generate optical flow and strain maps for the sample
    sample_path = os.path.join(opticalflow_path, sample)
    generate_optical_flow(sample_path)
