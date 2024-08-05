import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from CondiGan import Generator, Discriminator
from WindSpeedDataset import WindSpeedDataset
from experiment.validate import validate
from utils.dataset import partition
from utils.draw import plot_losses


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

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # 加载数据集
    train_data, val_data = partition(file_path=args.t_file, train_size=args.train_size)
    train_dataset = WindSpeedDataset(data=train_data)
    validate_dataset = WindSpeedDataset(data=val_data)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_data_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True)

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

    for epoch in range(args.epochs):

        for batch_idx, (real_data, condition) in enumerate(train_data_loader):
            real_data = real_data.to(args.device)
            condition = condition.to(args.device)
            z = torch.randn(real_data.size(0), args.noise_dim).to(args.device)

            # 训练判别器
            optimizer_D.zero_grad()
            fake_data = generator(z, condition)
            real_output = discriminator(real_data, condition)
            fake_output = discriminator(fake_data.detach(), condition)
            d_loss = criterion(real_output, torch.ones_like(real_output)) + criterion(fake_output,
                                                                                      torch.zeros_like(fake_output))
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_data, condition)
            g_loss = criterion(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            optimizer_G.step()

            # 打印损失信息
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], "
                      f"Batch [{batch_idx}/{len(train_data_loader)}], "
                      f"d_loss: {d_loss.item()}, "
                      f"g_loss: {g_loss.item()}")

        # 验证阶段
        avg_real_loss, avg_fake_loss, avg_total_loss = validate(generator, discriminator, validate_data_loader,
                                                                criterion, args)

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

    # 训练结束后绘制损失曲线
    plot_losses(epochs, {'Average Real Loss': real_losses,
                         'Average Fake Loss': fake_losses,
                         'Average Total Loss': total_losses}, 'epoch', 'avg_loss')

    return generator, discriminator
