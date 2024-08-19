import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import PathPatch
from matplotlib.path import Path


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


def plot_interpolation_comparison(full_data_all, generated_data_all, mask_all, time_step, feature_index, max_missing_len, save_file_name=None):
    # 获取指定特征的所有数据
    true_data = full_data_all[:, time_step, feature_index]  # 假设特征在最后一维
    generated_data = generated_data_all[:, time_step, feature_index]
    mask = mask_all[:, time_step, feature_index]

    # 特征名称映射
    feature_names = {0: 'windSpeed2m', 1: 'windSpeed10m'}
    feature_name = feature_names.get(feature_index, 'Unknown Feature')

    # 创建掩码
    missing_mask = mask == 0  # 标记缺失数据的位置
    valid_mask = mask == 1  # 标记有效数据的位置

    # 获取有效数据的时间步和对应数据
    time_steps_valid = np.where(valid_mask)[0]
    true_data_valid = true_data[valid_mask]
    generated_data_valid = generated_data[valid_mask]

    # 获取缺失数据的时间步和对应数据
    time_steps_missing = np.where(missing_mask)[0]
    true_data_missing = true_data[missing_mask]
    generated_data_missing = generated_data[missing_mask]

    # 创建绘图
    plt.figure(figsize=(14, 7))

    # 绘制真实数据（有效数据）
    plt.plot(time_steps_valid, true_data_valid, color='blue', linestyle='-', alpha=0.6, label='Real Data (Valid)')
    plt.plot(time_steps_valid, generated_data_valid, color='orange', linestyle='-', alpha=0.6,
             label='Generated Data (Valid)')

    # 绘制生成数据（缺失数据）
    if max_missing_len > 100:
        # 使用线条绘制生成数据（缺失数据）
        plt.plot(time_steps_missing, true_data_missing, color='red', linestyle='-', alpha=0.6,
                 label='Real Data (Missing)')
        plt.plot(time_steps_missing, generated_data_missing, color='green', linestyle='-', alpha=0.6,
                 label='Generated Data (Missing)')
    else:
        # 使用离散点绘制生成数据（缺失数据）
        plt.scatter(time_steps_missing, true_data_missing, color='red', label='Real Data (Missing)', s=50, marker='o',
                    zorder=4)
        plt.scatter(time_steps_missing, generated_data_missing, color='green', label='Generated Data (Missing)', s=50,
                    marker='x', zorder=5)

    # 添加图例和标签
    plt.legend()
    plt.title(f'Comparison of Interpolation for Feature {feature_name}')
    plt.xlabel('Time Steps (Hours)')
    plt.ylabel('Wind Speed (m/s)')
    plt.grid(True)

    if save_file_name:
        # 确保结果目录存在
        save_file_name = save_file_name + '.jpg'
        output_dir = './results-imgs/'
        os.makedirs(output_dir, exist_ok=True)
        # 构造保存路径
        file_path = os.path.join(output_dir, save_file_name)
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    else:
        plt.show()


if __name__ == '__main__':
    pass
