from torch import nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model import SRresNet, SRCNN
import time
import torch
from train import AverageMeter, crop_size
from nc_dataset import *
from nc_dataset import _normalize

# 模型参数
crop_size = 96
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 0.25  # 缩放比例
ngpu = 2  # GP数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # 测试集目录
    data_folder = "./dataset/EXTRACT_d3"

    # 预训练模型
    srresnet_checkpoint = "./results/110epoch_SRResNet.pth"   #110epoch_SRResNet.pth  70epoch_SRCNN.pth

    # # 加载模型SRResNet
    checkpoint = torch.load(srresnet_checkpoint)
    srresnet = SRresNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    srresnet.load_state_dict(checkpoint['model'])

    # 加载模型SRCNN
    # checkpoint = torch.load(srresnet_checkpoint)
    # srresnet = SRCNN()
    # srresnet.load_state_dict(checkpoint['model'])

    srresnet = srresnet.to(device)
    srresnet.eval()
    model = srresnet

    # 多GPU测试
    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(srresnet, device_ids=list(range(ngpu)))


    # 定制化数据加载器
    test_dataset = SRDataset(data_folder,
                             crop_size=crop_size,
                             scaling_factor=scaling_factor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                              pin_memory=True)

    # 记录每个样本 PSNR 和 SSIM值
    PSNRs = AverageMeter("PSNR")
    SSIMs = AverageMeter("SSIM")

    # 记录测试时间
    start = time.time()

    with torch.no_grad():
        # 逐批样本进行推理计算
        for i, (hr_tensor, lr_tensor) in enumerate(test_loader):
            # 数据移至默认设备
            lr_tensor = lr_tensor.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_tensor = hr_tensor.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

            # 前向传播.
            sr_tensor = model(lr_tensor)  # (1, 1, w, h), in [-1, 1]

            # 计算 PSNR 和 SSIM
            # sr_tensor_y = _normalize(sr_tensor).squeeze(
            #     0, 1)  # (w, h), in y-channel
            # hr_tensor_y = _normalize(hr_tensor).squeeze(
            #     0, 1) # (w, h), in y-channel

            sr_tensor_y = sr_tensor.squeeze(0,1)
            hr_tensor_y = hr_tensor.squeeze(0,1)
            psnr = peak_signal_noise_ratio(hr_tensor_y.cpu().numpy(), sr_tensor_y.cpu().numpy(), data_range=1.)
            ssim = structural_similarity(hr_tensor_y.cpu().numpy(), sr_tensor_y.cpu().numpy(), data_range=1., K2=0.01)
            PSNRs.update(psnr, lr_tensor.size(0))
            SSIMs.update(ssim, lr_tensor.size(0))

    # 输出平均PSNR和SSIM
    print('PSNR  {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM  {ssims.avg:.3f}'.format(ssims=SSIMs))
    print('平均单张样本用时  {:.3f} 秒'.format((time.time() - start) / len(test_dataset)))

print("\n")