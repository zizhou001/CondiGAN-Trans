import torch
from utils.losses import *


def validate(generator, discriminator, val_data_loader, criterion, args):
    generator.eval()
    discriminator.eval()
    total_real_loss = 0.0
    total_fake_loss = 0.0
    total_loss = 0.0
    total_batches = len(val_data_loader)

    # 遍历所有的验证数据批次，每批次获取真实数据和条件数据。
    with torch.no_grad():
        for val_batch_idx, (val_full_data, val_masked_data, val_condition, mask) in enumerate(val_data_loader):
            val_full_data = val_full_data.to(args.device)
            val_masked_data = val_masked_data.to(args.device)
            val_condition = val_condition.to(args.device)
            mask = mask.to(args.device)

            # 为每个批次生成随机噪声 z，这用于生成器生成伪造数据。
            z = torch.randn(val_full_data.size(0), args.noise_dim).to(args.device)

            # 调用生成器，输入随机噪声和条件数据，生成伪造数据
            val_fake_data, val_reconstructed_data = generator(z, val_condition, val_masked_data, mask=mask)

            # 将真实数据和生成的数据传入判别器，得到判别器对真实数据和伪造数据的输出
            val_real_output = discriminator(val_full_data, val_condition, mask)
            val_fake_output = discriminator(val_fake_data, val_condition, mask)

            # 计算真实数据的损失
            real_loss = criterion(val_real_output, torch.ones_like(val_real_output))
            total_real_loss += real_loss.item()

            # 计算伪造数据的损失
            fake_loss = criterion(val_fake_output, torch.ones_like(val_fake_output))
            total_fake_loss += fake_loss.item()

            # 计算重建损失
            recon_loss = reconstruction_loss(val_reconstructed_data, val_full_data, mask)

            # 计算插补损失
            interp_loss = interpolation_loss(val_fake_data, val_full_data, mask)

            # 总损失
            total_loss += (real_loss + fake_loss + recon_loss + interp_loss).item()

    # 计算验证损失的平均值
    avg_real_loss = total_real_loss / total_batches
    avg_fake_loss = total_fake_loss / total_batches
    avg_total_loss = total_loss / total_batches

    return avg_real_loss, avg_fake_loss, avg_total_loss


if __name__ == '__main__':
    pass
