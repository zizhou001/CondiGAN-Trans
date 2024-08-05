import matplotlib.pyplot as plt
import numpy as np


def plot_losses(x_data, y_data_dict, xlabel='Date', ylabel='Loss', title='Training Losses'):
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


if __name__ == '__main__':
    np.random.seed(0)  # 设置随机种子以获得可重复的结果
    batch_indices = np.arange(100)  # 横轴数据，0到99的批次索引
    real_losses = np.random.rand(100) * 1.5  # 模拟真实损失数据
    fake_losses = np.random.rand(100) * 1.5  # 模拟伪造损失数据
    total_losses = (real_losses + fake_losses) / 2  # 模拟总损失数据

    # 将数据放入字典
    y_data_dict = {
        'Average Real Loss': real_losses,
        'Average Fake Loss': fake_losses,
        'Average Total Loss': total_losses
    }

    # 绘图
    plot_losses(batch_indices, y_data_dict)
