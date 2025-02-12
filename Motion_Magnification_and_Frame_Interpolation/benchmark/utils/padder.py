
import torch.nn.functional as F


# 定义了一个名为 InputPadder 的类，用于对图像进行填充操作，以使其尺寸能够被给定的除数整除
class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor = 16):
        # 从输入的维度中提取图像的高度和宽度，并将其分配给实例变量 self.ht 和 self.wd
        self.ht, self.wd = dims[-2:]
        # 计算需要在高度方向上进行填充的像素数，以使图像的高度可被 divisor 整除
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        # 计算需要在宽度方向上进行填充的像素数，以使图像的宽度可被 divisor 整除
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        # 创建一个长度为 4 的列表 _pad，其中包含了填充宽度的左边界、右边界、填充高度的上边界和下边界
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    # 定义了一个 pad 方法，用于对输入进行填充操作。它接受任意个数的输入参数
    def pad(self, *inputs):
        # 对输入参数列表中的每个输入 x，使用 PyTorch 的 F.pad 函数对其进行填充操作，并使用 'replicate' 模式进行填充。最后，返回填充后的输入列表
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    # 定义了一个 unpad 方法，将填充后的图像裁剪为原始尺寸，去除填充的部分
    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]