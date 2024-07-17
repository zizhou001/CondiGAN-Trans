import torch


def validate(generator, discriminator, val_data_loader, criterion, args):
    generator.eval()
    discriminator.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_batch_idx, (val_real_data, val_condition) in enumerate(val_data_loader):
            val_real_data = val_real_data.to(args.device)
            val_condition = val_condition.to(args.device)
            z = torch.randn(val_real_data.size(0), args.noise_dim).to(args.device)
            val_fake_data = generator(val_real_data, z, val_condition)
            val_real_output = discriminator(val_real_data, val_condition)
            val_fake_output = discriminator(val_fake_data, val_condition)
            val_loss += (criterion(val_real_output, torch.ones_like(val_real_output)) +
                         criterion(val_fake_output, torch.zeros_like(val_fake_output))).item()

    # 计算验证损失的平均值
    val_loss /= len(val_data_loader)
    return val_loss



if __name__ == '__main__':
    pass
