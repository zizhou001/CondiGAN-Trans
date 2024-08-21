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


def plot_interpolation_comparison(full_data_all, generated_data_all, mask_all, time_step, feature_index,
                                  max_missing_len, save_file_name=None):
    # 获取指定特征的所有数据
    true_data = full_data_all[:, time_step, feature_index]  # 假设特征在最后一维
    generated_data = generated_data_all[:, time_step, feature_index]
    mask = mask_all[:, time_step, feature_index]

    # lh
    time_steps = np.arange(len(mask))
    time_steps_v = np.copy(time_steps).astype(float)
    time_steps_m = np.copy(time_steps).astype(float)
    true_data_v = np.full_like(true_data, np.nan, dtype=float)
    generated_data_v = np.full_like(generated_data, np.nan, dtype=float)
    ture_data_m = np.full_like(true_data, np.nan, dtype=float)
    generated_data_m = np.full_like(generated_data, np.nan, dtype=float)
    time_steps_v[mask == 0] = np.nan
    time_steps_m[mask == 1] = np.nan
    true_data_v[mask == 1] = true_data[mask == 1]
    generated_data_v[mask == 1] = generated_data[mask == 1]
    ture_data_m[mask == 0] = true_data[mask == 0]
    generated_data_m[mask == 0] = generated_data[mask == 0]

    # 特征名称映射
    feature_names = {0: 'windSpeed2m', 1: 'windSpeed10m'}
    feature_name = feature_names.get(feature_index, 'Unknown Feature')

    # 创建绘图
    plt.figure(figsize=(14, 7))

    # 绘制真实数据（有效数据）
    plt.plot(time_steps_v, true_data_v, color='#1f77b4', linestyle='-', alpha=0.6, label='Real Data')
    # plt.plot(time_steps_v, generated_data_v, color='orange', linestyle='-', alpha=0.6,
    #          label='Generated Data (Valid)')

    # 绘制生成数据（缺失数据）
    # if max_missing_len > 100:

    # 使用线条绘制生成数据（缺失数据）
    plt.plot(time_steps_m, ture_data_m, color='#2ca02c', linestyle='-', alpha=0.6,
             label='Real Data(Missing)')
    plt.plot(time_steps_m, generated_data_m, color='#d62728', linestyle='-', alpha=0.6,
             label='Generated Data')
    # else:
    #     # 使用离散点绘制生成数据（缺失数据）
    #     plt.scatter(time_steps_m, ture_data_m, color='orange', s=50, marker='o', label='Real Data(Missing)',
    #                 zorder=4)
    #     plt.scatter(time_steps_m, generated_data_m, color='red', s=50, label='Generated Data',
    #                 marker='x', zorder=5)

    # 添加图例和标签
    plt.legend()
    plt.grid(False)

    if save_file_name:
        # 确保结果目录存在
        save_file_name = save_file_name + '_' + feature_name + '.jpg'
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
