from scipy.signal import bilinear
from torch import nn
from model import SRresNet, SRCNN
import torch
import time
import xarray as xr
from torch.functional import F
import plot
from nc_dataset import _normalize
import random

random.seed(42)
torch.manual_seed(42)

# 测试数据
Path = './dataset/EXTRACT_d3/EXTRACT_wrfout_d03_2020-01-04_17_00_00.nc'

# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 0.25  # 缩放比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # 预训练模型
    srresnet_checkpoint = r"D:\WRFOUT_dataprocess\results\70epoch_SRCNN.pth"

    # 加载模型SRResNet
    # srresnet = SRresNet(large_kernel_size=large_kernel_size,
    #                     small_kernel_size=small_kernel_size,
    #                     n_channels=n_channels,
    #                     n_blocks=n_blocks,
    #                     scaling_factor=scaling_factor)
    # srresnet.load_state_dict(torch.load(srresnet_checkpoint)['model'])
    # srresnet = srresnet.to(device)

    # srresnet.eval()
    # model = srresnet
    # 加载模型SRCNN
    cnn = SRCNN()
    cnn.load_state_dict(torch.load(srresnet_checkpoint)['model'])
    cnn = cnn.to(device)

    cnn.eval()
    model = cnn


    # 加载图像
    _data = xr.open_dataset(Path, engine = 'netcdf4')
    var = _data.variables['wind_speed']
    _data = torch.from_numpy(var.data).float()
    _data = _normalize(_data)
    # # 从tensor中随机裁剪一个子块作为高分辨率tensor   hr_tensor
    # left = random.randint(1, _data.size()[1] - 48)
    # top = random.randint(1, _data.size()[2] - 48)
    # right = left + 48
    # bottom = top + 48
    # _data = _data[:, left:right, top:bottom]

    # 双三次插值上采样
    bicubic = F.interpolate(_data.unsqueeze(0), scale_factor=0.5, mode='bicubic', align_corners=False)
    bicubic = _normalize(bicubic)
    bilinear_ = F.interpolate(_data.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)
    bilinear_ = _normalize(bilinear_)
    plot.plot_grey(tensor=bilinear_.squeeze(0, 1), string='bilinear_30')
    plot.plot_grey(tensor=bicubic.squeeze(0, 1), string="BICUBIC_30")   #下采样结果



#--------------------------------------------------------------------
    # data__ = xr.open_dataset(f'{file_path}/{file}')
    # var = data__.variables["wind_speed"]
    # tensor__ = torch.from_numpy(var.data).float()
    #
    # # 从tensor中随机裁剪一个子块作为高分辨率tensor   hr_tensor
    # left = random.randint(1, tensor__.size()[1] - crop_size)
    # top = random.randint(1, tensor__.size()[2] - crop_size)
    # right = left + crop_size
    # bottom = top + crop_size
    # hr_tensor = tensor__[:, left:right, top:bottom]
    #
    # #双三次插值上采样
    # lr_tensor = F.interpolate(hr_tensor.unsqueeze(0), scale_factor=scaling_factor, mode="bicubic", align_corners=False)
    # lr_tensor = lr_tensor.squeeze(0)
    #
    # #归一化处理 [-1:1]
    # lr_tensor = _normalize(lr_tensor)
    # hr_tensor = _normalize(hr_tensor)
#--------------------------------------------------------------------


    # 记录时间
    start = time.time()

    # 转移数据至设备
    lr_img = bicubic.to(device)  # (1, 3, w, h ), imagenet-normed

    # 模型推理
    with torch.no_grad():
        sr_tensor = model(_data.unsqueeze(0)).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        plot.plot_grey(tensor=_data.squeeze(0), string="INPUT_30")                   #原始图像
        plot.plot_grey(tensor=sr_tensor.squeeze(0), string="SRRESNET_WRFOUT_30")     #输出图像

    print('用时  {:.3f} 秒'.format(time.time() - start))
