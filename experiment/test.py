import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from WindSpeedDataset import WindSpeedDataset
from utils.dataset import simulate_missing_data
from utils.draw import plot_imputation_results_single_feature


def interpolate(generator, args):
    # 读取数据
    data = pd.read_csv(args.i_file)

    # 模拟缺失
    mask = simulate_missing_data(data, column_names=args.column_names,
                                 max_missing_length=args.max_missing_length,
                                 missing_rate=args.missing_rate, missing_mode=args.missing_mode)

    # 加载数据
    dataset = WindSpeedDataset(data=data, mask=mask, columns=args.column_names)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    generator.eval()  # 切换到评估模式
    total_batches = len(data_loader)

    # 使用与训练时相同的归一化器
    scaler = dataset.scaler  # 获取归一化器

    # 准备绘图数据
    plot_data = {}

    true_values = []
    imputed_values = []
    mask_values = []

    with torch.no_grad():
        for val_batch_idx, (real_data, condition, mask) in enumerate(data_loader):
            real_data = real_data.to(args.device)
            condition = condition.to(args.device)
            mask = mask.to(args.device)

            # 为每个批次生成随机噪声 z，这用于生成器生成伪造数据。
            z = torch.randn(real_data.size(0), args.noise_dim).to(args.device)

            # 调用生成器，输入随机噪声和条件数据，生成插补后的数据
            imputed_data = generator(z, condition, mask)

            # 反归一化
            imputed_data_numpy = imputed_data.cpu().numpy()
            real_data_numpy = real_data.cpu().numpy()

            # 如果数据是三维的，通常需要将其展平成二维进行反归一化
            if imputed_data_numpy.ndim == 3:
                # 假设形状是 (batch_size, seq_length, features)，将其展平为 (batch_size * seq_length, features)
                num_samples, seq_length, num_features = imputed_data_numpy.shape
                imputed_data_numpy = imputed_data_numpy.reshape(-1, num_features)
                real_data_numpy = real_data_numpy.reshape(-1, num_features)

            imputed_data_original = scaler.inverse_transform(imputed_data_numpy)
            real_data_original = scaler.inverse_transform(real_data_numpy)

            # 如果需要，将数据恢复为原始的三维形状
            if imputed_data_numpy.shape[0] == num_samples * seq_length:
                imputed_data_original = imputed_data_original.reshape(num_samples, seq_length, num_features)
                real_data_original = real_data_original.reshape(num_samples, seq_length, num_features)

            # 记录数据用于绘图
            for feature_idx in range(num_features):
                feature_name = f'Feature_{feature_idx}'

                if feature_name not in plot_data:
                    plot_data[feature_name] = {
                        'original': [],
                        'imputed': [],
                        'mask': []
                    }

                for batch_idx in range(num_samples):
                    # 从真实数据中提取当前特征数据
                    original_data = real_data_original[batch_idx, :, feature_idx]
                    # 从掩码中提取当前特征数据的掩码
                    mask_data = mask.cpu().numpy()[batch_idx, :, feature_idx]
                    # 从插补数据中提取当前特征数据
                    imputed_data = imputed_data_original[batch_idx, :, feature_idx]

                    # 将插补数据累加到原始数据中，根据掩码更新
                    combined_data = np.where(mask_data == 1, imputed_data, original_data)

                    plot_data[feature_name]['original'].append(original_data)
                    plot_data[feature_name]['imputed'].append(combined_data)
                    plot_data[feature_name]['mask'].append(mask_data)

                    # 计算评价指标
                    true_values.extend(original_data[mask_data == 0])  # 真实值（未缺失的）
                    imputed_values.extend(imputed_data[mask_data == 1])  # 插补值（缺失的）
                    mask_values.extend(mask_data[mask_data == 1])  # 掩码值（缺失位置）

    true_values = np.array(true_values)
    imputed_values = np.array(imputed_values)

    # 确保 true_values 和 imputed_values 的长度一致
    if len(true_values) != len(imputed_values):
        print(f"Length mismatch: true_values({len(true_values)}) vs imputed_values({len(imputed_values)})")

    if len(true_values) > 0:
        mse = mean_squared_error(true_values, imputed_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, imputed_values)
        r2 = r2_score(true_values, imputed_values)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared Score (R2): {r2:.4f}")
    else:
        print("No valid data for evaluation.")

    # 绘制每个特征的插补结果
    for feature_name, data in plot_data.items():
        original_data = np.array(data['original'])
        imputed_data = np.array(data['imputed'])
        mask = np.array(data['mask'])

        plot_imputation_results_single_feature(
            original_data,
            imputed_data,
            mask,
            xlabel='Time Step',
            ylabel=feature_name,
            title=f'Imputation Results for {feature_name}'
        )


if __name__ == '__main__':
    pass
