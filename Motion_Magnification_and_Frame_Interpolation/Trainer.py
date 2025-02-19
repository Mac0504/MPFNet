import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from data.models_frame_interpolation.loss import *
from data.models_frame_interpolation.warplayer import warp

from config import *


class Model:
    def __init__(self, local_rank):
        # MODEL_CONFIG is a dictionary, and values can be accessed via MODEL_CONFIG['key']
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']  # (feature_extractor, flow_estimation)
        print('backbonetype:', backbonetype)  # Print the feature extractor function type
        print('multiscaletype:', multiscaletype)  # Print the flow estimation model type
        
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']  # Get the model architecture config
        # backbonecfg: {'embed_dims': [32, 64, 128, 256, 512], 'motion_dims': [0, 0, 0, 64, 128], ...}
        # multiscalecfg: {'scales': [4, 8, 16], 'hidden_dims': [128, 128], 'c': 32, ...}

        # Initialize the model network using the backbone and multi-scale flow estimation
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        # Get model name from configuration
        self.name = MODEL_CONFIG['LOGNAME']
        self.device()

        # Optimizer and Loss
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)  # AdamW optimizer with weight decay
        self.lap = LapLoss()  # LapLoss instance for loss calculation

        # If local_rank is not -1, use DistributedDataParallel (DDP) for multi-GPU training
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.net.train()  # Set model to training mode

    def eval(self):
        self.net.eval()  # Set model to evaluation mode

    def device(self):
        self.net.to(torch.device("cuda"))  # Move model to GPU

    # Load pre-trained model
    def load_model(self, name=None, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v  # Remove "module." from the parameter names
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k  # Exclude certain keys
            }

        if rank <= 0:
            if name is None:  # Use the default model name if none is provided
                name = self.name
            # Load the model weights from checkpoint
            self.net.load_state_dict(convert(torch.load(f'ckpt/{name}.pkl')))

    # Save model parameters to a checkpoint
    def save_model(self, rank=0):
        if rank == 0:  # Only save from rank 0 in distributed training
            torch.save(self.net.state_dict(), f'ckpt/{self.name}.pkl')

    @torch.no_grad()  # Disable gradient calculation during inference
    def hr_inference(self, img0, img1, TTA=False, down_scale=1.0, timestep=0.5, fast_TTA=False):
        '''
        TTA (Test Time Augmentation) indicates whether to use test-time data augmentation
        down_scale is the down-sampling factor for reducing input image resolution
        timestep is the time step used for interpolation
        fast_TTA indicates whether to use faster TTA (flipping images)
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]  # Split input image sequence into two parts: img0 and img1
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)  # Downscale images

            # Compute flow and mask using the model
            flow, mask = self.net.calculate_flow(imgs_down, timestep)

            # Upsample flow and mask back to the original size
            flow = F.interpolate(flow, scale_factor=1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor=1/down_scale, mode="bilinear", align_corners=False)

            # Extract features from img0 and img1
            af, _ = self.net.feature_bone(img0, img1)
            # Perform coarse warp and refine using the model
            pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred

        # Concatenate img0 and img1 along the channel dimension
        imgs = torch.cat((img0, img1), 1)
        
        if fast_TTA:
            # Flip images for augmentation
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)  # Combine original and flipped images
            preds = infer(input)  # Perform inference
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.  # Return flipped result averaged

        if not TTA:
            return infer(imgs)  # Normal inference without TTA
        else:
            return (infer(imgs) + infer(imgs.flip(2).flip(3)).flip(2).flip(3)) / 2  # Averaged TTA result

    @torch.no_grad()  # Disable gradient calculation during inference
    def inference(self, img0, img1, TTA=False, timestep=0.5, fast_TTA=False):
        # Concatenate img0 and img1 for inference
        imgs = torch.cat((img0, img1), 1)

        if fast_TTA:
            # Apply fast TTA (flip and combine both flipped and non-flipped images)
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            _, _, _, preds = self.net(input, timestep=timestep)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        _, _, _, pred = self.net(imgs, timestep=timestep)  # Perform inference
        if not TTA:
            return pred  # Normal inference without TTA
        else:
            _, _, _, pred2 = self.net(imgs.flip(2).flip(3), timestep=timestep)
            return (pred + pred2.flip(2).flip(3)) / 2  # Return averaged result from flipped image

    @torch.no_grad()
    def multi_inference(self, img0, img1, TTA=False, down_scale=1.0, time_list=[], fast_TTA=False):
        '''
        Run backbone once, get multiple frames at different timesteps
        Returns a list of frames at different timesteps
        '''
        assert len(time_list) > 0, 'Time_list should not be empty!'

        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]  # Split input into img0 and img1
            af, mf = self.net.feature_bone(img0, img1)  # Extract features

            # Optionally downscale the images
            imgs_down = None
            if down_scale != 1.0:
                imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
                afd, mfd = self.net.feature_bone(imgs_down[:, :3], imgs_down[:, 3:6])  # Extract downscaled features

            pred_list = []
            for timestep in time_list:
                # Calculate flow and mask for each timestep
                if imgs_down is None:
                    flow, mask = self.net.calculate_flow(imgs, timestep, af, mf)
                else:
                    flow, mask = self.net.calculate_flow(imgs_down, timestep, afd, mfd)
                    flow = F.interpolate(flow, scale_factor=1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
                    mask = F.interpolate(mask, scale_factor=1/down_scale, mode="bilinear", align_corners=False)
                
                pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
                pred_list.append(pred)

            return pred_list

        # Concatenate img0 and img1 along the channel dimension
        imgs = torch.cat((img0, img1), 1)

        if fast_TTA:
            # Apply fast TTA (flip images and infer)
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds_lst = infer(input)
            return [(preds_lst[i][0] + preds_lst[i][1].flip(1).flip(2))/2 for i in range(len(time_list))]

        preds = infer(imgs)
        if not TTA:
            return [preds[i][0] for i in range(len(time_list))]  # Normal inference without TTA
        else:
            flip_pred = infer(imgs.flip(2).flip(3))
            return [(preds[i][0
