import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class WindSpeedDataset(Dataset):
    def __init__(self, data, transform=None, seq_length=64, batch_size=32):
        # self.data = pd.read_csv(data, parse_dates=['date'])  # 解析日期列
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        self.data = data.copy()  # 使用传入的数据
        self.transform = transform
        self.seq_length = seq_length
        self.batch_size = batch_size

        # 创建归一化器
        self.scaler = MinMaxScaler()

        # 对需要归一化的列进行归一化处理
        self.data[['windSpeed3s', 'windSpeed2m']] = \
            self.scaler.fit_transform(self.data[['windSpeed3s', 'windSpeed2m']])
        # 列的索引
        self.windSpeed3s_column_index = self.data.columns.get_loc('windSpeed3s')
        self.windSpeed2m_column_index = self.data.columns.get_loc('windSpeed2m')



    def __len__(self):
        return len(self.data) - self.seq_length  # 确保可以提取完整序列

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取长度为 seq_length 的序列
        sample = self.data.iloc[idx:idx + self.seq_length]  # 提取 seq_length 长度的序列

        # 取风速数据
        time_series = sample[['windSpeed3s', 'windSpeed2m']].values.astype(
            np.float32)  # shape should be (seq_length, 2)

        # 提取小时信息
        hour = sample['date'].dt.hour.iloc[0]  # 提取第一个样本的小时信息

        # 提取风向信息
        wind_dir_3s = sample['windDir3s'].iloc[0]
        wind_dir_category = wind_direction_to_category(wind_dir_3s)

        # 构建时间条件one-hot编码(24维)
        time_condition_hourly = np.eye(24)[hour]

        # 提取日尺度数据（假设每天有24个小时）
        daily_start_idx = idx - (idx % (24 * 1))  # 确定每天数据的起始索引
        daily_sample = self.data.iloc[daily_start_idx:daily_start_idx + 24]  # 每天24小时的数据
        day_of_week = daily_sample['date'].iloc[0].dayofweek

        # 构建日尺度时间条件向量（7维，代表一周的7天）
        time_condition_daily = np.eye(7)[day_of_week]

        # 提取周尺度数据（假设每周有7天）
        weekly_start_idx = idx - (idx % (24 * 7))  # 确定每周数据的起始索引
        weekly_sample = self.data.iloc[weekly_start_idx:weekly_start_idx + 24 * 7]  # 每周7天的数据
        week_of_year = weekly_sample['date'].iloc[0].weekofyear

        # 构建周尺度时间条件向量（52维，代表一年的52周）
        time_condition_weekly = np.eye(52)[week_of_year - 1]  # 注意weekofyear从1开始，而索引从0开始

        # 构建风向条件one-hot编码（8维）
        wind_dir_conditions = np.eye(8)[wind_dir_category]  # 8个风向类别one-hot

        # 合并条件向量
        condition_vector = np.concatenate(
            (time_condition_hourly, time_condition_daily, time_condition_weekly, wind_dir_conditions))

        # 确保样本形状为 (seq_length, 2)
        sample_tensor = torch.from_numpy(time_series).float()

        # condition_tensor 确保是 (cond_dim,)  24+7+52+8=91
        condition_tensor = torch.from_numpy(condition_vector).float()


        return sample_tensor, condition_tensor  #


# 风向转换函数
def wind_direction_to_category(wind_dir):
    index = int((wind_dir + 22.5) // 45 % 8)
    return index  # 返回类别索引，而不是字符串
