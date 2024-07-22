import pandas as pd


def partition(file_path, train_size=0.8):
    data = pd.read_csv(file_path)
    train_length = int(len(data) * train_size)

    # 划分数据
    train_data = data[:train_length]
    val_data = data[train_length:]

    return train_data, val_data


def multiscale_divider(condition):
    # 分割条件向量
    hourly_condition = condition[:, :24]
    daily_condition = condition[:, 24:31]
    weekly_condition = condition[:, 31:83]
    wind_condition = condition[:, 83:]

    return hourly_condition, daily_condition, weekly_condition, wind_condition
