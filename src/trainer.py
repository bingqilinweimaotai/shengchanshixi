import math
import scipy
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'supervised_training_iter',
    'soc_adaptation_iter',
]


# ----------------------------------------------------------------------------------
# 工具类/函数
# ----------------------------------------------------------------------------------

class GaussianBlurLayer(nn.Module):
    """ 为4D张量添加高斯模糊
    该层接受形状为 {N, C, H, W} 的4D张量作为输入。
    高斯模糊将在给定的通道数(C)上分别执行。
    """

    def __init__(self, channels, kernel_size):
        """ 
        参数:
            channels (int): 输入张量的通道数
            kernel_size (int): 用于模糊的核大小
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)), 
            nn.Conv2d(channels, channels, self.kernel_size, 
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        参数:
            x (torch.Tensor): 输入的4D张量
        返回:
            torch.Tensor: 输入的模糊版本
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' 需要4D张量作为输入\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('在 \'GaussianBlurLayer\' 中，所需通道数 ({0}) 与'
                  '输入通道数 ({1}) 不同\n'.format(self.channels, x.shape[1]))
            exit()
            
        return self.op(x)
    
    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))

# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# MODNet 训练函数
# ----------------------------------------------------------------------------------

blurer = GaussianBlurLayer(1, 3).cuda()


def supervised_training_iter(
    modnet, optimizer, image, trimap, gt_matte,
    semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0):
    """ MODNet的有监督训练迭代
    此函数在有标签数据集中训练MODNet一个迭代。

    参数:
        modnet (torch.nn.Module): MODNet实例
        optimizer (torch.optim.Optimizer): 有监督训练的优化器
        image (torch.autograd.Variable): 输入的RGB图像
                                         其像素值应该被归一化
        trimap (torch.autograd.Variable): 用于计算损失的trimap
                                          其像素值可以是 0, 0.5, 或 1
                                          (前景=1, 背景=0, 未知=0.5)
        gt_matte (torch.autograd.Variable): 真实alpha遮罩
                                            其像素值在 [0, 1] 之间
        semantic_scale (float): 语义损失的缩放因子
                                注意: 请根据您的数据集调整
        detail_scale (float): 细节损失的缩放因子
                              注意: 请根据您的数据集调整
        matte_scale (float): 遮罩损失的缩放因子
                             注意: 请根据您的数据集调整
    
    返回:
        semantic_loss (torch.Tensor): 语义估计的损失 [低分辨率(LR)分支]
        detail_loss (torch.Tensor): 细节预测的损失 [高分辨率(HR)分支]
        matte_loss (torch.Tensor): 语义-细节融合的损失 [融合分支]

    示例:
        import torch
        from src.models.modnet import MODNet
        from src.trainer import supervised_training_iter

        bs = 16         # 批次大小
        lr = 0.01       # 学习率
        epochs = 40     # 总轮数

        modnet = torch.nn.DataParallel(MODNet()).cuda()
        optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

        dataloader = CREATE_YOUR_DATALOADER(bs)     # 注意: 请完成此函数

        for epoch in range(0, epochs):
            for idx, (image, trimap, gt_matte) in enumerate(dataloader):
                semantic_loss, detail_loss, matte_loss = \
                    supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)
            lr_scheduler.step()
    """

    global blurer

    # 将模型设置为训练模式并清除优化器
    modnet.train()
    optimizer.zero_grad()

    # 前向传播模型
    pred_semantic, pred_detail, pred_matte = modnet(image, False)

    # 从trimap计算边界掩码
    boundaries = (trimap < 0.5) + (trimap > 0.5)

    # 计算语义损失
    gt_semantic = F.interpolate(gt_matte, scale_factor=1/16, mode='bilinear')
    gt_semantic = blurer(gt_semantic)
    semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
    semantic_loss = semantic_scale * semantic_loss

    # 计算细节损失
    pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
    gt_detail = torch.where(boundaries, trimap, gt_matte)
    detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
    detail_loss = detail_scale * detail_loss

    # 计算遮罩损失
    pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
    matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)
    matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
        + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)
    matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
    matte_loss = matte_scale * matte_loss

    # 计算最终损失，反向传播损失，并更新模型
    loss = semantic_loss + detail_loss + matte_loss
    loss.backward()
    optimizer.step()

    # 用于测试
    return semantic_loss, detail_loss, matte_loss


def soc_adaptation_iter(
    modnet, backup_modnet, optimizer, image,
    soc_semantic_scale=100.0, soc_detail_scale=1.0):
    """ MODNet的自监督子目标一致性(SOC)适应迭代
    此函数在无标签数据集中微调MODNet一个迭代。
    注意SOC只能微调已收敛的MODNet，即在有标签数据集中训练过的MODNet。

    参数:
        modnet (torch.nn.Module): MODNet实例
        backup_modnet (torch.nn.Module): 训练好的MODNet的备份
        optimizer (torch.optim.Optimizer): 自监督SOC的优化器
        image (torch.autograd.Variable): 输入的RGB图像
                                         其像素值应该被归一化
        soc_semantic_scale (float): SOC语义损失的缩放因子
                                    注意: 请根据您的数据集调整
        soc_detail_scale (float): SOC细节损失的缩放因子
                                  注意: 请根据您的数据集调整
    
    返回:
        soc_semantic_loss (torch.Tensor): 语义SOC的损失
        soc_detail_loss (torch.Tensor): 细节SOC的损失

    示例:
        import copy
        import torch
        from src.models.modnet import MODNet
        from src.trainer import soc_adaptation_iter

        bs = 1          # 批次大小
        lr = 0.00001    # 学习率
        epochs = 10     # 总轮数

        modnet = torch.nn.DataParallel(MODNet()).cuda()
        modnet = LOAD_TRAINED_CKPT()    # 注意: 请完成此函数

        optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
        dataloader = CREATE_YOUR_DATALOADER(bs)     # 注意: 请完成此函数

        for epoch in range(0, epochs):
            backup_modnet = copy.deepcopy(modnet)
            for idx, (image) in enumerate(dataloader):
                soc_semantic_loss, soc_detail_loss = \
                    soc_adaptation_iter(modnet, backup_modnet, optimizer, image)
    """

    global blurer

    # 将备份模型设置为评估模式
    backup_modnet.eval()

    # 将主模型设置为训练模式并冻结其归一化层
    modnet.train()
    modnet.module.freeze_norm()

    # 清除优化器
    optimizer.zero_grad()

    # 前向传播主模型
    pred_semantic, pred_detail, pred_matte = modnet(image, False)

    # 前向传播备份模型
    with torch.no_grad():
        _, pred_backup_detail, pred_backup_matte = backup_modnet(image, False)

    # 从 `pred_matte` 和 `pred_semantic` 计算边界掩码
    pred_matte_fg = (pred_matte.detach() > 0.1).float()
    pred_semantic_fg = (pred_semantic.detach() > 0.1).float()
    pred_semantic_fg = F.interpolate(pred_semantic_fg, scale_factor=16, mode='bilinear')
    pred_fg = pred_matte_fg * pred_semantic_fg

    n, c, h, w = pred_matte.shape
    np_pred_fg = pred_fg.data.cpu().numpy()
    np_boundaries = np.zeros([n, c, h, w])
    for sdx in range(0, n):
        sample_np_boundaries = np_boundaries[sdx, 0, ...]
        sample_np_pred_fg = np_pred_fg[sdx, 0, ...]

        side = int((h + w) / 2 * 0.05)
        dilated = grey_dilation(sample_np_pred_fg, size=(side, side))
        eroded = grey_erosion(sample_np_pred_fg, size=(side, side))

        sample_np_boundaries[np.where(dilated - eroded != 0)] = 1
        np_boundaries[sdx, 0, ...] = sample_np_boundaries

    boundaries = torch.tensor(np_boundaries).float().cuda()

    # `pred_semantic` 和 `pred_matte` 之间的子目标一致性
    # 为 `pred_semantic` 生成伪真实标签
    downsampled_pred_matte = blurer(F.interpolate(pred_matte, scale_factor=1/16, mode='bilinear'))
    pseudo_gt_semantic = downsampled_pred_matte.detach()
    pseudo_gt_semantic = pseudo_gt_semantic * (pseudo_gt_semantic > 0.01).float()
    
    # 为 `pred_matte` 生成伪真实标签
    pseudo_gt_matte = pred_semantic.detach()
    pseudo_gt_matte = pseudo_gt_matte * (pseudo_gt_matte > 0.01).float()

    # 计算SOC语义损失
    soc_semantic_loss = F.mse_loss(pred_semantic, pseudo_gt_semantic) + F.mse_loss(downsampled_pred_matte, pseudo_gt_matte)
    soc_semantic_loss = soc_semantic_scale * torch.mean(soc_semantic_loss)

    # 注意: 使用我们论文中的公式计算以下损失有类似的结果
    # `pred_detail` 和 `pred_backup_detail` 之间的子目标一致性（仅在边界上）
    backup_detail_loss = boundaries * F.l1_loss(pred_detail, pred_backup_detail, reduction='none')
    backup_detail_loss = torch.sum(backup_detail_loss, dim=(1,2,3)) / torch.sum(boundaries, dim=(1,2,3))
    backup_detail_loss = torch.mean(backup_detail_loss)

    # `pred_matte` 和 `pred_backup_matte` 之间的子目标一致性（仅在边界上）
    backup_matte_loss = boundaries * F.l1_loss(pred_matte, pred_backup_matte, reduction='none')
    backup_matte_loss = torch.sum(backup_matte_loss, dim=(1,2,3)) / torch.sum(boundaries, dim=(1,2,3))
    backup_matte_loss = torch.mean(backup_matte_loss)

    soc_detail_loss = soc_detail_scale * (backup_detail_loss + backup_matte_loss)

    # 计算最终损失，反向传播损失，并更新模型
    loss = soc_semantic_loss + soc_detail_loss

    loss.backward()
    optimizer.step()

    return soc_semantic_loss, soc_detail_loss

# ----------------------------------------------------------------------------------
