from torch.utils.data import DataLoader
from WindSpeedDataset import WindSpeedDataset


def interpolate(generator, missing_data_file, args):
    dataset = WindSpeedDataset(missing_data_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    generator.eval()  # 切换到评估模式

    # 使用与训练时相同的归一化器
    scaler = dataset.scaler  # 获取归一化器

    for batch in dataloader:
        z = torch.randn(batch.size(0), args.noise_dim)  # 随机噪声
        condition = torch.randn(batch.size(0), args.cond_dim)  # 随机条件向量

        # 生成填补后的数据
        with torch.no_grad():
            imputed_data = generator(batch, z, condition)

        # 反归一化数据
        imputed_data_numpy = imputed_data.cpu().numpy()  # 转为numpy数组
        imputed_data_original = scaler.inverse_transform(imputed_data_numpy)  # 反归一化

        # 输出填补结果
        print("Imputed Data:")
        print(imputed_data_original)


if __name__ == '__main__':
    pass

