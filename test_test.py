import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from model import SRresNet
from nc_dataset import SRDataset
import os
import plot
import random

random.seed(42)
torch.manual_seed(42)

# 数据集参数
file_path = "D:\WRFOUT_dataprocess\dataset\EXTRACT_d1"  # 数据存放路径
crop_size = 48  # 高分辨率图像裁剪尺寸
scaling_factor = 0.25  # 缩放比例 0-1

# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量

# 学习参数
checkpoint = None  # 预训练模型路径，如果不存在则为None
batch_size = 8  # 批大小
start_epoch = 1  # 轮数起始位置
epochs = 20  # 迭代轮数
workers = 4  # 工作线程数
lr = 1e-3  # 学习率

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 2           # 用来运行的gpu数量

cudnn.benchmark = True # 对卷积进行加速

def main():
    # 预训练模型
    srresnet_checkpoint = r"D:\WRFOUT_dataprocess\results\70epoch_SRresNet.pth"

    # 加载模型SRResNet
    srresnet = SRresNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    srresnet.load_state_dict(torch.load(srresnet_checkpoint)['model'])
    srresnet = srresnet.to(device)

    model = srresnet

    # 迁移至默认设备进行训练
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # 定制化的dataloaders
    train_dataset = SRDataset(file_path, crop_size = crop_size, scaling_factor = scaling_factor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,
                             shuffle = True, num_workers = workers, pin_memory = True)

    for epoch in range(start_epoch, epochs + 1):
        print(f"==> training epoch {epoch}")
        model.train()  # 训练模式：允许使用批样本归一化

        loss_epoch = AverageMeter("MSE")  # 统计损失函数

        n_iter = len(train_loader)

        # 按批处理
        for i, (hr_tensor, lr_tensor) in enumerate(train_loader):
            # 数据移至默认设备进行训练
            hr_tensor = hr_tensor.to(device)  # (batch_size (N), 1, 12, 12), imagenet-normed 格式
            lr_tensor = lr_tensor.to(device)  # (batch_size (N), 1, 48, 48),  [-1, 1]格式

            # 前向传播
            sr_tensor = model(lr_tensor)

            # 计算损失
            loss = criterion(sr_tensor, hr_tensor).sqrt()

            # 记录损失值
            loss_epoch.update(loss.item(), lr_tensor.size(0))

            plot.plot_grey(tensor=sr_tensor[0].squeeze(0).detach().numpy(), string="SR_tensor_1")
            plot.plot_grey(tensor=hr_tensor[0].squeeze(0).detach().numpy(), string="HR_tensor_1")
            plot.plot_grey(tensor=lr_tensor[0].squeeze(0).detach().numpy(), string="LR_tensor_1")

            # 打印结果
            print("第 " + str(i) + " 个batch训练结束" + "  loss: " + str(loss_epoch.val))

            # 手动释放内存
            del lr_tensor, hr_tensor, sr_tensor
            break
        break


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    main()