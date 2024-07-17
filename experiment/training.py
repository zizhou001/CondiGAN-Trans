import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import DataLoader
from WindSpeedDataset import WindSpeedDataset
from CondiGan import Generator, Discriminator


def train(args, generator_saved_name, discriminator_saved_name):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 实例化生成器和判别器
    generator = Generator(d_model=args.d_model,
                          num_heads=args.num_heads,
                          num_layers=args.num_layers,
                          input_dim=args.input_dim,
                          noise_dim=args.noise_dim,
                          cond_emb_dim=args.cond_emb_dim,
                          noise_emb_dim=args.noise_emb_dim).to(args.device)
    discriminator = Discriminator(features_dim=args.features_dim,
                                  cond_dim=args.cond_dim,
                                  hidden_size=args.hidden_size, ).to(args.device)


    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr)

    # 引入文件
    dataset = WindSpeedDataset(csv_file=args.file_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 检查是否存在已有模型
    generator_path = './models/' + generator_saved_name
    discriminator_path = './models/' + discriminator_saved_name

    if os.path.exists(generator_path) and os.path.exists(discriminator_path):
        generator.load_state_dict(torch.load(generator_path))
        discriminator.load_state_dict(torch.load(discriminator_path))
        return generator, discriminator

    # 训练过程
    for epoch in range(args.epochs):
        for batch_idx, (real_data, condition) in enumerate(data_loader):
            real_data = real_data.to(args.device)  # 将数据移到设备
            condition = condition.to(args.device)  # 将条件信息移到设备

            # 生成随机噪声
            z = torch.randn(real_data.size(0), args.noise_dim).to(args.device)  # 随机噪声，形状为 (batch_size, noise_dim)

            # 训练判别器
            optimizer_D.zero_grad()
            fake_data = generator(real_data, z, condition)  # 输入 real_data, z 和 condition
            real_output = discriminator(real_data, condition)  # 使用 real_data 和 condition
            fake_output = discriminator(fake_data.detach(), condition)  # 使用 fake_data 和 condition，detach 避免计算梯度
            d_loss = criterion(real_output, torch.ones_like(real_output)) + criterion(fake_output,
                                                                                      torch.zeros_like(fake_output))
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_data, condition)  # 使用 fake_data 和 condition
            g_loss = criterion(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            optimizer_G.step()

            # 打印损失信息
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], "
                      f"Batch [{batch_idx}/{len(data_loader) // args.batch_size}],"
                      f" d_loss: {d_loss.item()},"
                      f" g_loss: {g_loss.item()}")

    # 保存模型
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    print("Models saved.")

    return generator, discriminator
