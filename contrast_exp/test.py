import numpy as np
import pandas as pd
from pypots.imputation import GRUD, BRITS, mrnn, gpvae, usgan, CSDI, nonstationary_transformer, TimesNet, SAITS
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch


# 读取数据
def load_data(path):
    data = pd.read_csv(path)
    return data


# 标准化数据
def standardize_data(data):
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    return data_standardized, scaler


# 创建数据加载器
def create_dataloader(data, batch_size=32):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


# 训练插补模型
def train_imputer(model, dataloader, epochs=100, device=None):
    if device is not None:
        model.set_device(device)
    model.fit(dataloader, epochs=epochs)
    return model


# 使用训练好的模型进行插补
def impute_data(model, dataloader, scaler, device=None):
    if device is not None:
        model.set_device(device)
    imputed_data = model.predict(dataloader)
    imputed_data = scaler.inverse_transform(imputed_data)
    return imputed_data


# 计算RMSE和MAE
def calculate_errors(real_data, imputed_data, missing_indices):
    real_data[missing_indices] = imputed_data[missing_indices]
    rmse = np.sqrt(mean_squared_error(real_data[~missing_indices], imputed_data[~missing_indices]))
    mae = mean_absolute_error(real_data[~missing_indices], imputed_data[~missing_indices])
    return rmse, mae


# 将结果输出到文件
def write_results_to_file(results, file_path):
    with open(file_path, 'a') as f:
        # 添加日期和时间信息
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Date and Time: {date_time}\n")

        for model_name, metrics in results.items():
            f.write(f"{model_name} - RMSE: {metrics['RMSE']:.3f} - MAE: {metrics['MAE']:.3f}\n")


# 划分数据集
def split_data(data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of ratios must be equal to 1."

    n_samples = len(data)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    test_size = n_samples - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


# 生成具有缺失值的数据
def generate_missing_data(df, column_names, missing_rate=0.1, max_missing_length=24, missing_mode='continuous'):
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
        for column_index in range(columns.shape[1]):
            num_segments = int(num_missing / max_missing_length)  # 分段数量
            current_missing_count = 0

            while current_missing_count < num_missing:
                start_index = np.random.randint(0, num_rows - max_missing_length + 1)
                end_index = min(start_index + max_missing_length, num_rows)
                segment_length = end_index - start_index

                # 确保剩余的缺失数量不会超过最大长度
                if num_missing - current_missing_count > segment_length:
                    mask[start_index:end_index, column_index] = 0
                    current_missing_count += segment_length
                else:
                    mask[start_index:start_index + (num_missing - current_missing_count), column_index] = 0
                    current_missing_count = num_missing

                # 确保掩码长度达到期望比例
                if np.sum(mask[:, column_index] == 0) >= num_missing:
                    break
    else:
        raise ValueError("Invalid missing_mode. Choose between 'random' and 'continuous'")

    # 将掩码应用到数据上
    masked_data = np.where(mask == 0, np.nan, columns)

    # 转换掩码矩阵为 tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    return masked_data, mask_tensor


# 比较不同模型的插补效果
def compare_models(data, models, parameters, epochs=100, batch_size=32, missing_mode='continuous', missing_rate=0.2,
                   device=None):
    results = {}

    for model_name, model_class in models.items():
        print(f"Processing model: {model_name}")

        # 标准化数据
        data_standardized, scaler = standardize_data(data.reshape(-1, 1))

        # 分割数据
        train_data, val_data, test_data = split_data(data_standardized)

        # 生成具有缺失值的数据
        train_data_missing, train_mask = generate_missing_data(train_data, missing_rate, missing_mode=missing_mode)
        test_data_missing, test_mask = generate_missing_data(test_data, missing_rate, missing_mode=missing_mode)

        # 创建数据加载器
        train_dataloader = create_dataloader(train_data_missing, batch_size=batch_size)
        test_dataloader = create_dataloader(test_data_missing, batch_size=batch_size)

        # 获取模型特定的参数
        model_params = parameters.get(model_name, {})

        # 训练模型
        imputer = model_class(**model_params, n_iter=epochs, batch_size=batch_size, r_seed=1)
        imputer = train_imputer(imputer, train_dataloader, epochs=epochs, device=device)

        # 使用模型插补
        test_imputed_data = impute_data(imputer, test_dataloader, scaler, device=device)

        # 计算RMSE和MAE
        rmse, mae = calculate_errors(test_data, test_imputed_data, test_mask)

        results[model_name] = {'RMSE': rmse, 'MAE': mae}

    return results


def main():
    # 检查是否有GPU可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义模型字典
    models = {
        "GRU-D": GRUD,
        "BRITS": BRITS,
        "M-RNN": mrnn.MRNN,
        "GP-VAE": gpvae.GPVAE,
        "US-GAN": usgan.US_GAN,
        "CSDI": CSDI,
        "Nonstationary Tr.": nonstationary_transformer.NonstationaryTr,
        "TimesNet": TimesNet,
        "SAITS": SAITS
    }

    # 参数配置字典
    parameters = {
        "GRU-D": {},
        "BRITS": {"n_layers": 1},
        "M-RNN": {"n_layers": 1, "n_hidden": 128},
        "GP-VAE": {"n_layers": 1, "n_hidden": 128},
        "US-GAN": {"n_layers": 1, "n_hidden": 128},
        "CSDI": {"n_layers": 1, "n_hidden": 128},
        "Nonstationary Tr.": {"n_layers": 1, "n_hidden": 128},
        "TimesNet": {"n_layers": 1, "n_hidden": 128},
        "SAITS": {"n_layers": 1, "n_hidden": 128}
    }

    # 加载数据
    data_path = 'contrast_data.csv'
    data = load_data(data_path)['windSpeed10m'].values

    # 比较模型
    results = compare_models(data, models, parameters, epochs=50, batch_size=64, device=device,
                             missing_mode='continuous', missing_rate=0.2)

    # 打印结果
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  MAE: {metrics['MAE']:.3f}")

    # 写入结果到文件
    file_path = 'contrast_results.txt'
    write_results_to_file(results, file_path)


if __name__ == "__main__":
    main()
