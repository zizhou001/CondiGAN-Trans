import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from CondiGan import Generator, Discriminator
from WindSpeedDataset import WindSpeedDataset
from experiment.validate import validate
from utils.dataset import partition, simulate_masked_data
from utils.draw import plot_show


# 定义 Wasserstein 损失
def wasserstein_loss(predictions, targets):
    return torch.mean(predictions * targets)


# 定义权重剪切
def weight_clip(model, clip_value):
    for param in model.parameters():
        param.data.clamp_(-clip_value, clip_value)


def train(args, generator_saved_name, discriminator_saved_name):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 实例化生成器和判别器
    generator = Generator(args.d_model, args.num_heads, args.num_layers, args.input_dim, args.seq_length,
                          args.cond_dim, args.noise_dim, args.noise_emb_dim, args.cond_emb_wind_dim,
                          args.features_dim, args.cond_emb_hourly_dim, args.cond_emb_daily_dim,
                          args.cond_emb_weekly_dim).to(args.device)
    discriminator = Discriminator(features_dim=args.features_dim,
                                  cond_dim=args.cond_dim,
                                  hidden_size=args.hidden_size).to(args.device)

    # 定义优化器
    optimizer_G = optim.RMSprop(generator.parameters(), lr=args.g_lr)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=args.d_lr)

    # 划分数据集
    train_data, val_data = partition(file_path=args.t_file, train_size=args.train_size)

    # 模拟缺失数据，获取掩码矩阵
    train_mask = simulate_masked_data(df=train_data, column_names=args.column_names,
                                      missing_rate=args.missing_rate,
                                      max_missing_length=args.max_missing_length,
                                      missing_mode=args.missing_mode)
    val_mask = simulate_masked_data(df=val_data, column_names=args.column_names,
                                    missing_rate=args.missing_rate,
                                    max_missing_length=args.max_missing_length,
                                    missing_mode=args.missing_mode)

    # 使用自定义数据集类加载数据
    train_dataset = WindSpeedDataset(data=train_data, columns=args.column_names, mask=train_mask)
    validate_dataset = WindSpeedDataset(data=val_data, columns=args.column_names, mask=val_mask)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_data_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

    # 检查是否存在已有模型
    generator_path = './checkpoints/' + generator_saved_name + '.checkpoint.pth'
    discriminator_path = './checkpoints/' + discriminator_saved_name + '.checkpoint.pth'

    if os.path.exists(generator_path) and os.path.exists(discriminator_path):
        generator.load_state_dict(torch.load(generator_path))
        discriminator.load_state_dict(torch.load(discriminator_path))
        return generator, discriminator

    # Early stopping参数
    best_val_loss = float('inf')  # 初始化为无穷大
    patience_counter = 0
    patience = args.patience  # 设置提前停止的耐心参数

    # 用于存储损失值
    real_losses = []
    fake_losses = []
    total_losses = []
    epochs = []

    # 权重剪切值
    clip_value = 0.01

    for epoch in range(args.epochs):

        for batch_idx, (full_data, masked_data, condition, mask) in enumerate(train_data_loader):
            full_data = full_data.to(args.device)
            condition = condition.to(args.device)
            masked_data = masked_data.to(args.device)
            mask = mask.to(args.device)
            z = torch.randn(full_data.size(0), args.noise_dim).to(args.device)

            # -----------------------------------训练判别器
            optimizer_D.zero_grad()
            fake_data, reconstructed_data = generator(z, condition, masked_data, mask=mask)
            real_output = discriminator(full_data, condition, mask=mask)
            fake_output = discriminator(fake_data.detach(), condition, mask=mask)
            d_loss = wasserstein_loss(torch.ones_like(real_output), real_output) + \
                     wasserstein_loss(-torch.ones_like(fake_output), fake_output)
            d_loss.backward()
            optimizer_D.step()

            # 权重剪切
            weight_clip(discriminator, clip_value)

            # -----------------------------------训练生成器
            optimizer_G.zero_grad()
            fake_data, reconstructed_data = generator(z, condition, masked_data, mask=mask)
            fake_output = discriminator(fake_data, condition, mask=mask)
            g_loss = wasserstein_loss(torch.ones_like(fake_output), fake_output)

            # 计算重建损失
            reconstruction_loss = nn.L1Loss(reduction='none')(reconstructed_data, full_data)
            reconstruction_loss = (reconstruction_loss * mask).sum() / mask.sum()
            g_total_loss = g_loss + reconstruction_loss

            g_total_loss.backward()
            optimizer_G.step()

            # 打印损失信息
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], "
                      f"Batch [{batch_idx}/{len(train_data_loader)}], "
                      f"d_loss: {d_loss.item()}, "
                      f"g_loss: {g_loss.item()}, "
                      f"reconstruction_loss: {reconstruction_loss.item()}")

        # 验证阶段
        avg_real_loss, avg_fake_loss, avg_total_loss = validate(generator, discriminator, validate_data_loader,
                                                                wasserstein_loss, args)

        real_losses.append(avg_real_loss)
        fake_losses.append(avg_fake_loss)
        total_losses.append(avg_total_loss)
        epochs.append(epoch)

        print(f"Validation Loss after epoch {epoch} | "
              f"avg_total_loss: {avg_total_loss}, "
              f"avg_real_loss: {avg_real_loss}, "
              f"avg_fake_loss: {avg_fake_loss}")

        # Early stopping逻辑
        if avg_total_loss < best_val_loss:
            best_val_loss = avg_total_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(generator.state_dict(), generator_path)
            torch.save(discriminator.state_dict(), discriminator_path)
            print("Models saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return generator, discriminator
