import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from data.models_frame_interpolation.loss import *
from data.models_frame_interpolation.warplayer import warp

from config import *

    
class Model:
    def __init__(self, local_rank):
        # MODEL_CONFIG是一个字典，通过MODEL_CONFIG['键']来取值
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE'] # (feature_extractor, flow_estimation)
        print('backbonetype:', backbonetype) # backbonetype: <function feature_extractor at 0x000001C85E2DF378>
        print('multiscaletype:', multiscaletype) # multiscaletype: <class 'model.flow_estimation.MultiScaleFlow'>
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH'] # backbonecfg 和 multiscalecfg 是从 MODEL_CONFIG 中获取的模型架构配置
        # backbonecfg: {'embed_dims': [32, 64, 128, 256, 512], 'motion_dims': [0, 0, 0, 64, 128], 'num_heads': [8, 16], 'mlp_ratios': [4, 4],
        # 'qkv_bias': True, 'norm_layer': functools.partial(<class 'torch.nn.modules.normalization.LayerNorm'>, eps=1e-06), 'depths': [2, 2, 2, 4, 4], 'window_sizes': [7, 7]}
        # print('backbonecfg:', backbonecfg)
        # multiscalecfg: {'embed_dims': [32, 64, 128, 256, 512], 'motion_dims': [0, 0, 0, 64, 128], 'depths': [2, 2, 2, 4, 4], 'num_heads': [8, 16],
        # 'window_sizes': [7, 7], 'scales': [4, 8, 16], 'hidden_dims': [128, 128], 'c': 32}
        # print('multiscalecfg:', multiscalecfg)

        # 通过使用这些配置初始化了 self.net，即模型网络
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        # 从 MODEL_CONFIG 中获取的日志名称
        self.name = MODEL_CONFIG['LOGNAME']
        self.device()

        # train
        # self.net.parameters()返回了self.net网络模型中的所有可学习参数，这些参数包括权重和偏置等模型参数
        # 创建了优化器 self.optimG，使用 AdamW 优化算法，并将其应用于参数（self.net.parameters()），这些参数将被AdamW优化器用于计算梯度并更新网络模型的参数
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        # 创建了损失函数 self.lap，使用 LapLoss 类的实例化对象
        self.lap = LapLoss()
        # 如果 local_rank 不等于 -1，则使用分布式数据并行方法（DDP）对模型进行封装
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.net.train() # 将模型设置为训练模式

    def eval(self):
        self.net.eval() # 将模型设置为评估模式

    def device(self):
        self.net.to(torch.device("cuda"))

    # 加载预训练模型
    def load_model(self, name=None, rank=0):
        # 加载模型参数时，使用 convert 函数对参数进行转换，去除键名中的 "module." 前缀，并排除特定键
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        if rank <= 0 :
            if name is None: # 如果没有指定模型名称 name，则使用默认的 self.name
                name = self.name
            # 加载的模型参数来自于路径 'ckpt/{name}.pkl'
            self.net.load_state_dict(convert(torch.load(f'ckpt/{name}.pkl')))

    # 保存模型参数
    def save_model(self, rank=0):
        if rank == 0: # 只有在 rank 等于 0 时才会执行保存操作
            # 模型参数会保存在路径 'ckpt/{self.name}.pkl' 下
            torch.save(self.net.state_dict(),f'ckpt/{self.name}.pkl')

    @torch.no_grad() # @torch.no_grad() 是一个装饰器，用于指定在函数执行期间不计算梯度。这意味着函数中的所有操作都不会影响模型的梯度
    # 用于高分辨率图像推断的函数
    def hr_inference(self, img0, img1, TTA = False, down_scale = 1.0, timestep = 0.5, fast_TTA = False):
        '''
        TTA（Test Time Augmentation）表示是否使用测试时数据增强
        down_scale 是一个降采样因子，用于降低输入图像的分辨率
        timestep 是时间步长
        fast_TTA 表示是否使用快速 TTA
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6] # 将输入的图像序列imgs分为两个部分，img0和img1，分别包含前三个通道和后三个通道的像素值
            # 使用双线性插值方法将输入图像序列imgs进行下采样，缩小尺寸为原来的down_scale倍，并将结果存储在imgs_down中
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
            # 将下采样后的图像序列imgs_down和时间步长timestep作为输入，通过神经网络模型self.net计算光流和遮罩。flow表示计算得到的光流，mask表示计算得到的遮罩
            flow, mask = self.net.calculate_flow(imgs_down, timestep)

            # 对计算得到的光流flow进行插值操作，将其尺寸恢复到原始尺寸，并乘以(1/down_scale)进行缩放。这样可以将光流恢复到与原始图像相同的尺寸
            # align_corners参数指定了插值操作是否应该在角点对齐
            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            # 对计算得到的遮罩mask进行插值操作，将其尺寸恢复到原始尺寸
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)

            # 将img0和img1作为输入，通过神经网络模型self.net提取特征。结果存储在af中
            af, _ = self.net.feature_bone(img0, img1)
            # 将输入图像序列imgs、提取的特征af、光流flow和遮罩mask作为输入，通过神经网络模型self.net进行粗糙的图像变形和细化，并得到推理结果pred
            pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred

        # 将img0和img1在通道维度上进行连接，得到新的图像序列imgs
        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            # 对输入图像序列imgs进行上下翻转和左右翻转操作，得到新的图像序列imgs_
            imgs_ = imgs.flip(2).flip(3)
            # 将imgs和imgs_在批次维度上进行连接，得到新的输入input
            input = torch.cat((imgs, imgs_), 0)
            # 得到推理结果preds
            preds = infer(input)
            # 将推理结果preds的第一个元素和第二个元素进行上下翻转和左右翻转操作，然后相加。最后，对结果进行维度扩展，并除以2，得到最终的推理结果并返回
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        if TTA == False:
            return infer(imgs)
        else:
            return (infer(imgs) + infer(imgs.flip(2).flip(3)).flip(2).flip(3)) / 2

    @torch.no_grad()
    def inference(self, img0, img1, TTA = False, timestep = 0.5, fast_TTA = False):
        # 将img0和img1在通道维度上进行拼接，得到一个新的张量imgs。这是为了将两个图像作为输入传递给模型
        imgs = torch.cat((img0, img1), 1)
        '''
        Noting: return BxCxHxW
        '''
        # 如果使用测试时数据增强，则对输入图像进行翻转并将其与原始图像一起输入模型，最后将推理结果进行翻转和融合
        if fast_TTA:
            # 对输入图像序列imgs进行上下翻转和左右翻转操作，得到新的图像序列imgs_
            # 这是测试时数据增强的一种方式，通过对输入图像进行翻转来获得更多的样本
            imgs_ = imgs.flip(2).flip(3)
            # 将imgs和imgs_在批次维度上进行连接，得到新的输入input，将原始图像和经过翻转的图像一起作为输入传递给模型
            input = torch.cat((imgs, imgs_), 0)
            # 对输入input进行推理，得到推理结果preds
            _, _, _, preds = self.net(input, timestep=timestep)
            # 将推理结果preds的第一个元素和第二个元素进行上下翻转和左右翻转操作，然后相加。最后，对结果进行维度扩展，并除以2，得到最终的推理结果并返回
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        # 在不使用测试时数据增强的情况下，直接将imgs作为输入传递给模型进行推理，得到推理结果pred
        _, _, _, pred = self.net(imgs, timestep=timestep)
        if TTA == False:
            return pred
        else:
            _, _, _, pred2 = self.net(imgs.flip(2).flip(3), timestep=timestep)
            return (pred + pred2.flip(2).flip(3)) / 2

    @torch.no_grad()
    # 定义了一个名为multi_inference的函数，它接受两个输入图像img0和img1，以及一些可选参数。TTA表示是否进行测试时增强（Test-Time Augmentation）
    # down_scale表示是否进行缩放，time_list表示不同时间步的列表，fast_TTA表示是否使用快速测试时增强
    def multi_inference(self, img0, img1, TTA = False, down_scale = 1.0, time_list=[], fast_TTA = False):
        '''
        Run backbone once, get multi frames at different timesteps
        Noting: return a list of [CxHxW]
        '''
        assert len(time_list) > 0, 'Time_list should not be empty!'
        def infer(imgs): # torch.Size([2, 6, 480, 640])
            img0, img1 = imgs[:, :3], imgs[:, 3:6] # 第一个维度上选择所有元素（即所有图像），第二个维度上选择前三个通道，得到img0
            af, mf = self.net.feature_bone(img0, img1) # 获得特征af和mf
            imgs_down = None
            if down_scale != 1.0:
                imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
                afd, mfd = self.net.feature_bone(imgs_down[:, :3], imgs_down[:, 3:6]) # 获得缩放后的特征afd和mfd

            pred_list = []
            for timestep in time_list:
                if imgs_down is None:
                    flow, mask = self.net.calculate_flow(imgs, timestep, af, mf)
                else:
                    flow, mask = self.net.calculate_flow(imgs_down, timestep, afd, mfd)
                    flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
                    mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)
                
                pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
                pred_list.append(pred)

            return pred_list

        imgs = torch.cat((img0, img1), 1) # torch.Size([1, 6, 480, 640])
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds_lst = infer(input)
            return [(preds_lst[i][0] + preds_lst[i][1].flip(1).flip(2))/2 for i in range(len(time_list))]

        preds = infer(imgs)
        if TTA is False:
            return [preds[i][0] for i in range(len(time_list))]
        else:
            flip_pred = infer(imgs.flip(2).flip(3))
            return [(preds[i][0] + flip_pred[i][0].flip(1).flip(2))/2 for i in range(len(time_list))]
    
    def update(self, imgs, gt, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            flow, mask, merged, pred = self.net(imgs)
            loss_l1 = (self.lap(pred, gt)).mean()

            for merge in merged:
                loss_l1 += (self.lap(merge, gt)).mean() * 0.5

            self.optimG.zero_grad()
            loss_l1.backward()
            self.optimG.step()
            return pred, loss_l1
        else: 
            with torch.no_grad():
                flow, mask, merged, pred = self.net(imgs)
                return pred, 0
