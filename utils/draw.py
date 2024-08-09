import matplotlib.pyplot as plt
import numpy as np


def plot_show(x_data, y_data_dict, xlabel='Date', ylabel='Loss', title='Training Losses'):
    """
    绘制损失曲线

    参数:
    - x_data: 横轴数据（如 batch_idx）
    - y_data_dict: 字典，键为标签，值为对应的纵轴数据列表
    - xlabel: 横轴标签
    - ylabel: 纵轴标签
    - title: 图形标题
    """
    plt.figure(figsize=(10, 6))

    # 如果 x_data 为空或 None，生成默认的 x_data
    if x_data is None or len(x_data) == 0:
        # 使用 y_data_dict 中任意一组数据的长度生成默认的 x_data
        # 这里假设所有 y_data 的长度相同
        example_length = len(next(iter(y_data_dict.values())))
        x_data = np.arange(example_length)

    # 绘制每一组纵轴数据
    for label, y_data in y_data_dict.items():
        # 确保 y_data 长度与 x_data 长度一致
        if len(x_data) != len(y_data):
            raise ValueError(
                f"Length of x_data ({len(x_data)}) does not match length of y_data for label '{label}' ({len(y_data)})")
        plt.plot(x_data, y_data, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_imputation_results_single_feature(original_data, imputed_data, mask, xlabel='Time', ylabel='Value',
                                           title='Imputation Results for Single Feature'):
    """
    绘制插补结果，仅处理单个特征

    参数:
    - original_data: 原始数据，形状为 (seq_length, 1)
    - imputed_data: 插补后的数据，形状为 (seq_length, 1)
    - mask: 掩码，形状为 (seq_length, 1)，缺失的数据点为 1，其他为 0
    - xlabel: 横轴标签
    - ylabel: 纵轴标签
    - title: 图形标题
    """
    plt.figure(figsize=(12, 8))

    # 单个特征的数据
    original_feature = original_data.flatten()
    imputed_feature = imputed_data.flatten()
    mask_feature = mask.flatten()

    # 绘制原始数据（完整数据）
    plt.plot(original_feature, label='Original Data', color='gray', linestyle='--')

    # 绘制插补数据
    plt.plot(imputed_feature, label='Imputed Data', color='blue')

    # 标记缺失值和插补值
    missing_indices = np.where(mask_feature == 1)[0]  # 掩码中为1的地方是缺失数据
    plt.scatter(missing_indices, original_feature[missing_indices], color='red', marker='x',
                label='Missing Original Data')
    plt.scatter(missing_indices, imputed_feature[missing_indices], color='green', marker='o', label='Imputed Points')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    original_data = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0]
    ])

    # 插补后的数据
    imputed_data = np.array([
        [1.0],
        [2.5],
        [4.0],
        [5.0],
        [4.0]
    ])

    # 掩码，其中1表示缺失
    mask = np.array([
        [0],
        [1],
        [0],
        [0],
        [1]
    ])

    # 调用绘图函数
    plot_imputation_results_single_feature(original_data, imputed_data, mask)
