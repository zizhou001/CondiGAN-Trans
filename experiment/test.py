from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from WindSpeedDataset import WindSpeedDataset
from utils.dataset import simulate_masked_data
from utils.draw import *


def interpolate(generator, args, remark):
    # 读取数据
    data = pd.read_csv(args.i_file)

    # 模拟缺失
    mask = simulate_masked_data(data, column_names=args.column_names,
                                max_missing_length=args.max_missing_length,
                                missing_rate=args.missing_rate, missing_mode=args.missing_mode)

    # 加载数据
    dataset = WindSpeedDataset(data=data, mask=mask, columns=args.column_names)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    generator.eval()  # 切换到评估模式
    total_batches = len(data_loader)

    # 使用与训练时相同的归一化器
    scaler = dataset.scaler  # 获取归一化器

    total_mse = 0.0
    total_rmse = 0.0
    total_valid_points = 0
    all_full_data = []
    all_generated_data = []
    all_mask = []

    with torch.no_grad():
        for val_batch_idx, (full_data, masked_data, condition, mask) in enumerate(data_loader):
            full_data = full_data.to(args.device)
            condition = condition.to(args.device)
            masked_data = masked_data.to(args.device)
            mask = mask.to(args.device)

            # 为每个批次生成随机噪声 z，这用于生成器生成伪造数据。
            z = torch.randn(full_data.size(0), args.noise_dim).to(args.device)

            # 调用生成器，输入随机噪声和条件数据，生成插补后的数据
            generated_data, imputed_data = generator(z, condition, masked_data, mask=mask)

            # 反归一化
            generated_data_numpy = generated_data.cpu().numpy()
            real_data_numpy = full_data.cpu().numpy()
            mask_numpy = mask.cpu().numpy()

            # 如果数据是三维的，通常需要将其展平成二维进行反归一化
            if generated_data_numpy.ndim == 3:
                # 假设形状是 (batch_size, seq_length, features)，将其展平为 (batch_size * seq_length, features)
                num_samples, seq_length, num_features = generated_data_numpy.shape
                generated_data_numpy = generated_data_numpy.reshape(-1, num_features)
                real_data_numpy = real_data_numpy.reshape(-1, num_features)
                mask_numpy = mask_numpy.reshape(-1, num_features)

            generated_original = scaler.inverse_transform(generated_data_numpy)
            real_data_original = scaler.inverse_transform(real_data_numpy)

            # 如果需要，将数据恢复为原始的三维形状
            if generated_data_numpy.shape[0] == num_samples * seq_length:
                generated_original = generated_original.reshape(num_samples, seq_length, num_features)
                real_data_original = real_data_original.reshape(num_samples, seq_length, num_features)
                mask_numpy = mask_numpy.reshape(num_samples, seq_length, num_features)

            # 记录数据
            all_full_data.append(real_data_original)
            all_generated_data.append(generated_original)
            all_mask.append(mask_numpy)

    full_data_all = np.concatenate(all_full_data, axis=0)
    generated_data_all = np.concatenate(all_generated_data, axis=0)
    mask_all = np.concatenate(all_mask, axis=0)

    # 计算全局平均 MSE 和 RMSE
    missing_mask_all = mask_all == 0
    if np.sum(missing_mask_all) > 0:
        # 对缺失数据位置计算 MSE、RMSE 和 MAE
        mse = np.mean((generated_data_all[missing_mask_all] - full_data_all[missing_mask_all]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(generated_data_all[missing_mask_all] - full_data_all[missing_mask_all]))
        avg_mse = mse
        avg_rmse = rmse
        avg_mae = mae
    else:
        avg_mse = 0
        avg_rmse = 0
        avg_mae = 0

    print(f'Average MSE: {avg_mse}')
    print(f'Average RMSE: {avg_rmse}')
    print(f'Average MAE: {avg_mae}')

    # 获取当前日期和时间
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 将结果写入文件
    with open('metrics.txt', 'a') as file:
        file.write(f"{current_datetime}    {remark}    "
                   f"MAE={avg_mae:.3f}    "
                   f"MSE={avg_mse:.3f}    "
                   f"RMSE={avg_rmse:.3f}\n")

    plot_interpolation_comparison(full_data_all, generated_data_all, mask_all, 0, 0)
    plot_interpolation_comparison(full_data_all, generated_data_all, mask_all, 0, 1)


if __name__ == '__main__':
    pass
