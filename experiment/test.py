from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from WindSpeedDataset import WindSpeedDataset
from utils.dataset import simulate_masked_data
from utils.draw import *


def interpolate(generator, args, remark, test_data):

    # 模拟缺失
    mask = simulate_masked_data(test_data, column_names=args.column_names,
                                max_missing_length=args.max_missing_length,
                                missing_rate=args.missing_rate, missing_mode=args.missing_mode)

    # 加载数据
    dataset = WindSpeedDataset(data=test_data, mask=mask, columns=args.column_names, seq_length=args.seq_length, batch_size=args.batch_size)
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

    num_features = full_data_all.shape[2]  # 特征数量

    # 计算每个特征的 MSE、RMSE 和 MAE
    mse_per_feature = np.zeros(num_features)
    rmse_per_feature = np.zeros(num_features)
    mae_per_feature = np.zeros(num_features)

    missing_mask_all = mask_all == 0
    if np.sum(missing_mask_all) > 0:
        for feature_idx in range(num_features):
            feature_mask = missing_mask_all[:, :, feature_idx].flatten()
            if np.sum(feature_mask) > 0:
                mse_per_feature[feature_idx] = np.mean((generated_data_all[:, :, feature_idx].flatten()[feature_mask] -
                                                        full_data_all[:, :, feature_idx].flatten()[feature_mask]) ** 2)
                rmse_per_feature[feature_idx] = np.sqrt(mse_per_feature[feature_idx])
                mae_per_feature[feature_idx] = np.mean(np.abs(
                    generated_data_all[:, :, feature_idx].flatten()[feature_mask] -
                    full_data_all[:, :, feature_idx].flatten()[feature_mask]))

        avg_mse = np.mean(mse_per_feature)
        avg_rmse = np.mean(rmse_per_feature)
        avg_mae = np.mean(mae_per_feature)
    else:
        mse_per_feature.fill(0)
        rmse_per_feature.fill(0)
        mae_per_feature.fill(0)
        avg_mse = 0
        avg_rmse = 0
        avg_mae = 0

    # print(f'Average MSE: {avg_mse}')
    # print(f'Average RMSE: {avg_rmse}')
    # print(f'Average MAE: {avg_mae}')

    # 打印每个特征的误差
    # for i in range(num_features):
    #     print(
    #         f'Feature {i} - MSE: {mse_per_feature[i]:.3f}, RMSE: {rmse_per_feature[i]:.3f}, MAE: {mae_per_feature[i]:.3f}')



    # 获取当前日期和时间
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 将结果写入文件
    with open('metrics.txt', 'a', encoding='utf-8') as file:
        file.write(f"{current_datetime}    {remark}    "
                   f"MAE={avg_mae:.3f}    "
                   f"MSE={avg_mse:.3f}    "
                   f"RMSE={avg_rmse:.3f}\n")

        # 打印每个特征的误差
        for i in range(num_features):
            file.write(
                f"\tFeature {i}: MSE={mse_per_feature[i]:.3f}     RMSE={rmse_per_feature[i]:.3f}     MAE={mae_per_feature[i]:.3f}\n"
            )

    plot_interpolation_comparison(full_data_all, generated_data_all, mask_all, time_step=int(args.seq_length // 2),
                                  feature_index=0,
                                  max_missing_len=args.max_missing_length, save_file_name=remark)
    plot_interpolation_comparison(full_data_all, generated_data_all, mask_all, time_step=int(args.seq_length // 2),
                                  feature_index=1,
                                  max_missing_len=args.max_missing_length, save_file_name=remark)


def test_mask_counts(mask):
    """
    测试 mask 数据中的 0 和 1 的个数，支持 NumPy 数组和 PyTorch 张量。

    参数:
    mask (np.ndarray or torch.Tensor): 需要测试的掩码数据

    输出:
    None
    """
    # 处理 NumPy 数组
    if isinstance(mask, np.ndarray):
        num_zeros = np.sum(mask == 0)
        num_ones = np.sum(mask == 1)

    # 处理 PyTorch 张量
    elif isinstance(mask, torch.Tensor):
        mask_numpy = mask.cpu().numpy()  # 将张量转换为 NumPy 数组
        num_zeros = np.sum(mask_numpy == 0)
        num_ones = np.sum(mask_numpy == 1)

    # 不支持其他数据类型
    else:
        raise TypeError("mask 应该是一个 NumPy 数组或 PyTorch 张量")

    # 打印结果
    print(f"Mask 数据中 0 的个数: {num_zeros}")
    print(f"Mask 数据中 1 的个数: {num_ones}")


if __name__ == '__main__':
    pass
