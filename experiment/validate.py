from torch.utils.data import DataLoader
from WindSpeedDataset import WindSpeedDataset


def interpolate(generator, missing_data_file, args):
    dataset = WindSpeedDataset(missing_data_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    generator.eval()  # 切换到评估模式

    for batch in dataloader:
        z = torch.randn(batch.size(0), args.noise_dim)  # 随机噪声
        condition = torch.randn(batch.size(0), args.cond_dim)  # 随机条件向量

        # 生成填补后的数据
        with torch.no_grad():
            imputed_data = generator(batch, z, condition)

        # 输出填补结果
        print("Imputed Data:")
        print(imputed_data.numpy())

