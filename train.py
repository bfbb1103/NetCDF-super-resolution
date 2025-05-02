import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from model import SRresNet, SRCNN
from nc_dataset import SRDataset
import os

# 数据集参数
file_path = "D:\WRFOUT_dataprocess\dataset\EXTRACT_d2"  # 数据存放路径
crop_size = 48  # 高分辨率图像裁剪尺寸
scaling_factor = 0.25  # 缩放比例 0-1

# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量

# 学习参数
checkpoint = r"D:\WRFOUT_dataprocess\results\30epoch_SRCNN.pth"  # 预训练模型路径，如果不存在则为None
batch_size = 8  # 批大小
start_epoch = 1  # 轮数起始位置
epochs = 70  # 迭代轮数
workers = 4  # 工作线程数
lr = 1e-3  # 学习率

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 2           # 用来运行的gpu数量

cudnn.benchmark = True # 对卷积进行加速

writer = SummaryWriter() # 实时监控     使用命令 tensorboard --logdir runs  进行查看

def main():
    global start_epoch, writer, checkpoint
    print("==> train start...")
    #模型初始化
    # model = SRresNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
    #                  n_channels=n_channels, scaling_factor=scaling_factor).to(device)
    model = SRCNN(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size).to(device)
    print(f"modle initialize successfully! current model is {model.name}")
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 迁移至默认设备进行训练
    model = model.to(device)
    criterion = nn.MSELoss().to(device)
    # 加载预训练模型
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))

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
            hr_tensor = hr_tensor.to(device)  # (batch_size (N), 1, 48, 48)
            lr_tensor = lr_tensor.to(device)  # (batch_size (N), 1, 12, 12)

            # 前向传播
            sr_tensor = model(lr_tensor)

            # 计算损失
            mse_loss = criterion(sr_tensor, hr_tensor)
            loss = mse_loss.sqrt()

            # 后向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新模型
            optimizer.step()

            # 记录损失值
            loss_epoch.update(loss.item(), lr_tensor.size(0))

            # 监控图像变化
            if i == (n_iter - 2):
                writer.add_image(f'{model}/epoch_' + str(epoch) + '_1',
                                 make_grid(lr_tensor[:4, :, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image(f'{model}/epoch_' + str(epoch) + '_2',
                                 make_grid(sr_tensor[:4, :, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image(f'{model}/epoch_' + str(epoch) + '_3',
                                 make_grid(hr_tensor[:4, :, :, :].cpu(), nrow=4, normalize=True), epoch)

            # 打印结果
            print("第 " + str(i) + " 个batch训练结束" + "  loss: " + str(loss_epoch.val))

        # 手动释放内存
        del lr_tensor, hr_tensor, sr_tensor

        # 监控损失值变化
        writer.add_scalar('SRResNet/MSE_Loss', loss_epoch.val, epoch)

    # 保存预训练模型
    torch.save({
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, f'results/{epochs}epoch_{model.name}.pth')
    print("model save successfully!")

    # 训练结束关闭监控
    writer.close()

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