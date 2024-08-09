import pandas as pd
from torch.utils.data import DataLoader
from WindSpeedDataset import WindSpeedDataset
import torch
import numpy as np
from utils.draw import plot_losses
from utils.dataset import simulate_missing_data
from sklearn.metrics import mean_squared_error, mean_absolute_error


def interpolate(generator, args):
    # 读取数据
    data = pd.read_csv(args.i_file)

    # 模拟缺失
    data_missing, mask = simulate_missing_data(data, column_names=args.column_names,
                                               max_missing_length=args.max_missing_length,
                                               missing_rate=args.missing_rate, missing_mode=args.missing_mode)

    # 加载数据
    dataset = WindSpeedDataset(data_missing)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    generator.eval()  # 切换到评估模式
    total_batches = len(data_loader)

    # 使用与训练时相同的归一化器
    scaler = dataset.scaler  # 获取归一化器

    # bug
    windSpeed3s_idx = 0
    windSpeed2m_idx = 1

    all_real_windSpeed3s = []
    all_imputed_windSpeed3s = []
    all_real_windSpeed2m = []
    all_imputed_windSpeed2m = []

    with torch.no_grad():
        for val_batch_idx, (real_data, condition) in enumerate(data_loader):
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

            original_windSpeed3s = real_data_original[:, :, windSpeed3s_idx]
            imputed_windSpeed3s = imputed_data_original[:, :, windSpeed3s_idx]
            original_windSpeed2m = real_data_original[:, :, windSpeed2m_idx]
            imputed_windSpeed2m = imputed_data_original[:, :, windSpeed2m_idx]

            # 累积数据
            all_real_windSpeed3s.append(original_windSpeed3s[:, 0])
            all_imputed_windSpeed3s.append(imputed_windSpeed3s[:, 0])
            all_real_windSpeed2m.append(original_windSpeed2m[:, 0])
            all_imputed_windSpeed2m.append(imputed_windSpeed2m[:, 0])

    all_real_windSpeed3s = np.concatenate(all_real_windSpeed3s, axis=0)
    all_imputed_windSpeed3s = np.concatenate(all_imputed_windSpeed3s, axis=0)
    all_real_windSpeed2m = np.concatenate(all_real_windSpeed2m, axis=0)
    all_imputed_windSpeed2m = np.concatenate(all_imputed_windSpeed2m, axis=0)

    mse_windSpeed3s = mean_squared_error(all_real_windSpeed3s, all_imputed_windSpeed3s)
    rmse_windSpeed3s = np.sqrt(mse_windSpeed3s)
    mae_windSpeed3s = mean_absolute_error(all_real_windSpeed3s, all_imputed_windSpeed3s)

    mse_windSpeed2m = mean_squared_error(all_real_windSpeed2m, all_imputed_windSpeed2m)
    rmse_windSpeed2m = np.sqrt(mse_windSpeed2m)
    mae_windSpeed2m = mean_absolute_error(all_real_windSpeed2m, all_imputed_windSpeed2m)

    print('--------------test set error statistics------------------------')
    print(f'[windSpeed3s]mse:{mse_windSpeed3s} | '
          f'rmse: {rmse_windSpeed3s} | '
          f'mae: {mae_windSpeed3s}')
    print(f'[windSpeed2m]mse:{mse_windSpeed2m} | '
          f'rmse: {rmse_windSpeed2m} | '
          f'mae: {mae_windSpeed2m}')

    plot_losses(x_data=[], y_data_dict={'imputed-windSpeed3s': all_imputed_windSpeed3s,
                                        'real-windSpeed3s': all_real_windSpeed3s}, title='windSpeed3s')

    plot_losses(x_data=[], y_data_dict={'imputed-windSpeed2m': all_imputed_windSpeed2m,
                                        'real-windSpeed2m': all_real_windSpeed2m}, title='windSpeed2m')


if __name__ == '__main__':
    pass
