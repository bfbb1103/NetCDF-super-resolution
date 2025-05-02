import torch
from torch.utils.data import Dataset
import os
import xarray as xr
import random
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

#将nc文件转为tensor, (channels, H, W)
def NetCDFTransforms(file_path, file, crop_size, scaling_factor):
    data__ = xr.open_dataset(f'{file_path}/{file}')
    var = data__.variables["wind_speed"]
    tensor__ = torch.from_numpy(var.data).float()

    # 从tensor中随机裁剪一个子块作为高分辨率tensor   hr_tensor
    left = random.randint(1, tensor__.size()[1] - crop_size)
    top = random.randint(1, tensor__.size()[2] - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    hr_tensor = tensor__[:, left:right, top:bottom]

    #双三次插值下采样
    lr_tensor = F.interpolate(hr_tensor.unsqueeze(0), scale_factor=scaling_factor, mode="bicubic", align_corners=False)
    lr_tensor = lr_tensor.squeeze(0)

    #归一化处理 [-1:1]
    lr_tensor = _normalize(lr_tensor)
    hr_tensor = _normalize(hr_tensor)

    return hr_tensor, lr_tensor

class SRDataset(Dataset):
    """
        crop_size: 裁剪尺寸  从nc生成的tensor中裁剪的尺寸
        scaling_factor: 缩小比例  图片分辨率缩小倍数
        二者相乘为整数
        比如 scaling_factor = 0.25， crop_size = 96, 96 * 0.25= 24
    """
    def __init__(self, file_path, crop_size, scaling_factor):
        self.file_path = file_path
        self.files = os.listdir(f'{file_path}')
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor

    def __len__(self):
        return self.files.__len__()

    def __getitem__(self, item):
        """
            返回高分辨率和低分辨率tensor
        :param item: 索引值
        :return: 高tensor，低tensor
        """
        file = self.files[item]
        hr_tensor, lr_tensor = NetCDFTransforms(self.file_path, file, self.crop_size, self.scaling_factor)
        return hr_tensor, lr_tensor

def _normalize(tensor):
    min = torch.min(tensor)
    max = torch.max(tensor)
    res = (tensor - min) / (max - min)
    return res*2 - 1