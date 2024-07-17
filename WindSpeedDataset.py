import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class WindSpeedDataset(Dataset):
    def __init__(self, csv_file, transform=None, seq_length=64, batch_size=32):
        self.data = pd.read_csv(csv_file, parse_dates=['date'])  # 解析日期列
        self.transform = transform
        self.seq_length = seq_length
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data) - self.seq_length  # 确保可以提取完整序列

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取长度为 seq_length 的序列
        sample = self.data.iloc[idx:idx + self.seq_length]  # 提取 seq_length 长度的序列

        # 取风速数据
        time_series = sample[['windSpeed3s', 'windSpeed2m']].values.astype(np.float32)  # shape should be (seq_length, 2)

        # 提取小时信息
        hour = sample['date'].dt.hour.iloc[0]  # 提取第一个样本的小时信息

        wind_dir_3s = sample['windDir3s'].iloc[0]  # 提取第一个样本的风向
        wind_dir_category = wind_direction_to_category(wind_dir_3s)  # 转换风向为类别

        # 构建时间条件one-hot编码(24维)
        time_condition = np.eye(24)[hour]

        # 构建风向条件one-hot编码（8维）
        wind_dir_conditions = np.eye(8)[wind_dir_category]  # 8个风向类别one-hot

        # 合并条件向量
        condition_vector = np.concatenate((time_condition, wind_dir_conditions))  # shape (32,)

        # 确保样本形状为 (seq_length, 2)
        sample_tensor = torch.from_numpy(time_series).float()  # shape (64, 2)

        # 生成 real_data
        # 这里不再使用 unsqueeze 和 expand，确保真实数据形状为 (seq_length, 2)
        real_data = sample_tensor

        # condition_tensor 需要确保是 (batch_size, cond_dim)
        condition_tensor = torch.from_numpy(condition_vector).float()  # shape (32, 32)

        return real_data, condition_tensor  # 确保返回的形状是 (32, 64, 2) 和 (32, 32)


# 风向转换函数
def wind_direction_to_category(wind_dir):
    index = int((wind_dir + 22.5) // 45 % 8)
    return index  # 返回类别索引，而不是字符串
