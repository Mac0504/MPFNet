# Video Motion Magnification and Frame Interpolation

## Step 1. From the original samples, obtain the sequence from the onset frame to the apex frame.

👉	get_select_frames.py

code_file = r'D:\0.Malcolm\1.Micro-expression database\CASME2\CASME2-coding.xlsx'

img_source_path = r'D:\0.Malcolm\1.Micro-expression database\CASME2\CASME2_preprocessed_Li Xiaobai\CASME2_preprocessed_Li Xiaobai\Cropped'

img_dst_path_onset2apex = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex'

get_onset_to_apex(code_file, img_source_path, img_dst_path_onset2apex)

## Step 2. Input the sequence from the starting frame to the peak frame and apply batch resizing to the image sequence, with the resizing coefficient as the hyperparameter.

👉	motion_magnification.py

pics_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex'

pics_mag_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_onset2apex_mag' + str(amplification)

mag_pics(model_path, pics_path, pics_mag_path, amplification)

## Step 3. From the resized frame sequence, obtain the starting frame and peak frame.

👉	get_select_frames.py

img_key_frames_path = r'data\examples\CASME2_pics\CASME2_pics_cropped_key_frames'

get_onset_and_apex_amplified(code_file, img_onset_path, img_apex_path, img_offset_path, img_key_frames_path)

## Step 4. Based on the starting frame and peak frame, perform frame interpolation, with the number of interpolation frames as the hyperparameter.

👉	frame_interpolation_Nx.py

    # Perform frame interpolation based on the first and last images.

pic1 = r'data\examples\CASME2_pics\CASME2_pics_cropped_key_frames\sub01\EP19_06f\reg_img36.jpg'

pic2 = r'data\examples\CASME2_pics\CASME2_pics_cropped_key_frames\sub01\EP19_06f\reg_img36.jpg'

interpolation(args, pic1, pic2)

The storage path for the interpolated frame sequence is:

D:\0.Malcolm\2.Projects\10.Motion_Magnification_and_Frame_Interpolation\data\examples\CASME2_pics\CASME2_pics_cropped_interpolation_frames

Original engineering code for the motion magnification algorithm: 

[T. Oh, R. Jaroensri, C. Kim, M. Elgharib, F. Durand, W. Freeman, W. Matusik "Learning-based Video Motion Magnification" arXiv preprint arXiv:1804.02684 (2018)](https://people.csail.mit.edu/tiam/deepmag/) in PyTorch.
