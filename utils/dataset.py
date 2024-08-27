import pandas as pd
import numpy as np
import torch


# def partition(file_path, train_size=0.8):
#     data = pd.read_csv(file_path)
#     train_length = int(len(data) * train_size)
#
#     # 划分数据
#     train_data = data[:train_length]
#     val_data = data[train_length:]
#
#     return train_data, val_data


def partition(file_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 读取数据
    data = pd.read_csv(file_path)

    # 计算各部分的长度
    total_length = len(data)
    train_length = int(total_length * train_ratio)
    val_length = int(total_length * val_ratio)

    # 划分数据
    train_data = data[:train_length]
    val_data = data[train_length:train_length + val_length]
    test_data = data[train_length + val_length:]

    return train_data, val_data, test_data


def multiscale_divider(condition):
    # 分割条件向量
    hourly_condition = condition[:, :24]
    daily_condition = condition[:, 24:31]
    weekly_condition = condition[:, 31:83]
    wind_condition = condition[:, 83:]

    return hourly_condition, daily_condition, weekly_condition, wind_condition


'''
def simulate_masked_data(df, column_names, missing_rate=0.1, max_missing_length=24, missing_mode='continuous'):
    """
    模拟缺失数据的掩码，并返回掩码矩阵。

    :param df: 输入数据框
    :param column_names: 需要插补的列名
    :param missing_rate: 缺失数据的比例
    :param max_missing_length: 最大缺失长度（用于连续缺失模式）
    :param missing_mode: 缺失模式，'random' 表示离散随机缺失，'continuous' 表示连续长序列缺失
    :return: 掩码矩阵
        0 代表缺失数据的位置，即数据在这些位置是缺失的。
        1 代表数据存在的位置，即这些位置的数据是有效的。
    """

    df_copy = df.copy()
    num_rows = len(df_copy)
    columns = df_copy[column_names].values

    # 初始化掩码矩阵
    mask = np.ones(columns.shape, dtype=np.float32)

    if missing_mode == 'random':
        # 随机缺失
        num_missing = int(num_rows * missing_rate)
        missing_indices = np.random.choice(num_rows, num_missing, replace=False)
        mask[missing_indices] = 0
    elif missing_mode == 'continuous':
        # 连续长序列缺失
        num_missing = int(num_rows * missing_rate)
        if max_missing_length > num_missing:
            max_missing_length = num_missing
        for column_index in range(columns.shape[1]):
            num_segments = max(int(num_missing / max_missing_length), 1)
            centering = (num_segments == 1)
            for _ in range(num_segments):
                if centering:
                    # Center the segment
                    middle_index = num_rows // 2
                    start_index = max(0, middle_index - max_missing_length // 2)
                    end_index = min(start_index + max_missing_length, num_rows)

                    # Adjust start index to fit within bounds
                    if end_index - start_index < max_missing_length:
                        start_index = max(0, end_index - max_missing_length)

                else:
                    # Randomly select start index
                    start_index = np.random.randint(0, num_rows - max_missing_length + 1)
                    end_index = start_index + max_missing_length

                # Ensure indices are integers
                start_index = int(start_index)
                end_index = int(end_index)

                # Ensure indices are within bounds and do not exceed the number of rows
                start_index = max(0, start_index)
                end_index = min(num_rows, end_index)

                # Apply the mask
                mask[start_index:end_index, column_index] = 0

                # Ensure mask length reaches expected proportion
                if np.sum(mask[:, column_index] == 0) >= num_missing:
                    break
    else:
        raise ValueError("Invalid missing_mode. Choose between 'random' and 'continuous'")

    # 转换掩码矩阵为 tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    return mask_tensor
'''

'''
def simulate_masked_data(df, column_names, missing_rate=0.1, max_missing_length=24, missing_mode='continuous'):
    """
    模拟缺失数据的掩码，并返回掩码矩阵。连续缺失片段之间至少相隔max_missing_length

    :param df: 输入数据框
    :param column_names: 需要插补的列名
    :param missing_rate: 缺失数据的比例
    :param max_missing_length: 最大缺失长度（用于连续缺失模式）
    :param missing_mode: 缺失模式，'random' 表示离散随机缺失，'continuous' 表示连续长序列缺失
    :return: 掩码矩阵
        0 代表缺失数据的位置，即数据在这些位置是缺失的。
        1 代表数据存在的位置，即这些位置的数据是有效的。
    """

    df_copy = df.copy()
    num_rows = len(df_copy)
    columns = df_copy[column_names].values

    # 初始化掩码矩阵
    mask = np.ones(columns.shape, dtype=np.float32)

    if missing_mode == 'random':
        # 随机缺失
        num_missing = int(num_rows * missing_rate)
        missing_indices = np.random.choice(num_rows, num_missing, replace=False)
        mask[missing_indices] = 0
    elif missing_mode == 'continuous':
        # 连续长序列缺失
        num_missing = int(num_rows * missing_rate)
        if max_missing_length > num_missing:
            max_missing_length = num_missing

        for column_index in range(columns.shape[1]):
            num_segments = max(1, int(num_missing / max_missing_length))
            segment_start_indices = []

            # 确保至少有一个缺失片段
            if num_segments == 0:
                num_segments = 1
                segment_start_indices.append(0)
            else:
                # 首个缺失片段的开始位置
                start_index = np.random.randint(0, num_rows - max_missing_length)
                segment_start_indices.append(start_index)

                for _ in range(num_segments - 1):
                    # 计算下一个片段的起始位置
                    min_start_index = segment_start_indices[-1] + max_missing_length
                    if min_start_index + max_missing_length >= num_rows:
                        break  # 没有足够空间生成下一个片段
                    start_index = np.random.randint(min_start_index, num_rows - max_missing_length)
                    segment_start_indices.append(start_index)

            # 如果最后一个片段可能超出数据范围，则调整
            if segment_start_indices[-1] + max_missing_length > num_rows:
                segment_start_indices[-1] = num_rows - max_missing_length

            # 填充缺失区域
            for start_index in segment_start_indices:
                end_index = start_index + max_missing_length
                # 确保 start_index 和 end_index 是整数
                start_index = int(start_index)
                end_index = int(end_index)
                mask[start_index:end_index, column_index] = 0

            # 确保掩码矩阵符合预期的缺失比例
            if np.sum(mask[:, column_index] == 0) >= num_missing:
                break
    else:
        raise ValueError("Invalid missing_mode. Choose between 'random' and 'continuous'")

    # 转换掩码矩阵为 tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    return mask_tensor
'''



def simulate_masked_data(df, column_names, missing_rate=0.1, max_missing_length=24, missing_mode='continuous'):
    """
    模拟缺失数据的掩码，并返回掩码矩阵。连续缺失片段之间至少相隔min_gap

    :param df: 输入数据框
    :param column_names: 需要插补的列名
    :param missing_rate: 缺失数据的比例
    :param max_missing_length: 最大缺失长度（用于连续缺失模式）
    :param missing_mode: 缺失模式，'random' 表示离散随机缺失，'continuous' 表示连续长序列缺失
    :return: 掩码矩阵
        0 代表缺失数据的位置，即数据在这些位置是缺失的。
        1 代表数据存在的位置，即这些位置的数据是有效的。
    """
    min_gap = 100
    df_copy = df.copy()
    num_rows = len(df_copy)
    columns = df_copy[column_names].values

    # 初始化掩码矩阵
    mask = np.ones(columns.shape, dtype=np.float32)

    if missing_mode == 'random':
        # 随机缺失
        num_missing = int(num_rows * missing_rate)
        missing_indices = np.random.choice(num_rows, num_missing, replace=False)
        mask[missing_indices] = 0

    elif missing_mode == 'continuous':
        # 连续长序列缺失
        num_missing = int(num_rows * missing_rate)
        if max_missing_length > num_missing:
            max_missing_length = num_missing

        # 计算测试集范围（前20%和后20%）
        test_start_index = int(num_rows * 0.2)
        test_end_index = int(num_rows * 0.8)

        # 计算可用区域（去掉测试集区域）
        available_start = test_start_index
        available_end = test_end_index - max_missing_length

        for column_index in range(columns.shape[1]):
            num_segments = max(1, int((available_end - available_start + 1) / (max_missing_length + min_gap)))
            segment_start_indices = []
            current_start = available_start

            # 生成均匀分布的缺失片段起始位置
            while current_start + max_missing_length <= available_end:
                # 确保生成的起始位置在有效区域内
                max_start = min(available_end, current_start + (max_missing_length + min_gap))
                if max_start <= current_start:
                    break
                start_index = np.random.randint(current_start, max_start)
                segment_start_indices.append(start_index)
                current_start = start_index + max_missing_length + min_gap

            # 填充缺失区域
            for start_index in segment_start_indices:
                end_index = start_index + max_missing_length
                # 确保 start_index 和 end_index 是整数
                start_index = int(start_index)
                end_index = int(end_index)
                mask[start_index:end_index, column_index] = 0

            # 确保掩码矩阵符合预期的缺失比例
            if np.sum(mask[:, column_index] == 0) >= num_missing:
                break

    else:
        raise ValueError("Invalid missing_mode. Choose between 'random' and 'continuous'")

    # 转换掩码矩阵为 tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    return mask_tensor
