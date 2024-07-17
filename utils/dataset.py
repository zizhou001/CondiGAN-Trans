import pandas as pd


def partition(file_path, train_size=0.8):
    data = pd.read_csv(file_path)
    train_length = int(len(data) * train_size)

    # 划分数据
    train_data = data[:train_length]
    val_data = data[train_length:]

    return train_data, val_data
