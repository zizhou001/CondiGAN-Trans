from torch import nn
import torch


def discriminator_loss(real_output, fake_output):
    """
    计算判别器的损失。
    """
    criterion = nn.BCEWithLogitsLoss()  # 更适合GAN的损失函数
    real_loss = criterion(real_output, torch.ones_like(real_output))
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss


def generator_loss(fake_output, reconstructed_data, fake_data, full_data, mask):
    """
    计算生成器的损失，包括判别器损失、重建损失和插补损失。
    """
    # 计算生成器的判别器损失
    criterion = nn.BCEWithLogitsLoss()  # 使用 BCE 适用于GAN的对抗损失
    g_loss = criterion(fake_output, torch.ones_like(fake_output))

    # 计算重建损失
    recon_loss = reconstruction_loss(reconstructed_data, full_data, mask)

    # 计算插补损失
    interp_loss = interpolation_loss(fake_data, full_data, mask)

    # 综合损失
    total_loss = g_loss + recon_loss + interp_loss
    return total_loss



def reconstruction_loss(reconstructed_data, real_data, mask):
    # 计算每个元素的L1损失
    loss = nn.L1Loss(reduction='none')(reconstructed_data, real_data)
    # 使用掩码将重建损失限制在正常的数据上
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)  # 添加小常数避免除以零
    return loss


def interpolation_loss(generated_data, real_data, mask):
    # 计算每个元素的L2损失
    loss = nn.MSELoss(reduction='none')(generated_data, real_data)
    # 使用掩码将插补损失限制在缺失的数据上
    loss = (loss * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)  # 添加小常数避免除以零
    return loss


def wasserstein_loss(predictions, targets):
    return torch.mean(predictions * targets)


# 定义权重剪切
def weight_clip(model, clip_value):
    for param in model.parameters():
        param.data.clamp_(-clip_value, clip_value)


def custom_loss_function(reconstructed_data, real_data, generated_data, mask, lambda_reconstruction=1.0,
                         lambda_interpolation=1.0):
    # 计算重建损失
    rec_loss = reconstruction_loss(reconstructed_data, real_data, mask)
    # 计算插补损失
    interp_loss = interpolation_loss(generated_data, real_data, mask)
    # 综合损失
    total_loss = lambda_reconstruction * rec_loss + lambda_interpolation * interp_loss
    return total_loss


if __name__ == '__main__':
    pass
