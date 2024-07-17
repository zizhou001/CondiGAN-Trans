# bug
1. training中，文件名应该抽取出来
2. main总，考虑验证和训练的关系

下面是用于测试的示例代码。

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class WindDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, parse_dates=['date'])
        self.data = self.data.fillna(0)  # 替换缺失值为0（或其他策略）

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row[['windSpeed3s', 'windDir3s', 'windSpeed2m', 'windDir2m', 'windSpeed10m', 'windDir10m', 'temperature']].values
        return torch.tensor(features, dtype=torch.float32)

# 创建数据集和数据加载器
csv_file = 'path_to_your_data.csv'  # 替换为实际的 CSV 文件路径
dataset = WindDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # 根据需要设置 batch_size

# 假设您的生成器和条件设置如下
generator = Generator()  # 替换为实际模型初始化
generator.load_state_dict(torch.load('path_to_your_generator_model.pth'))
generator.eval()

# 测试填补缺失值
for batch in dataloader:
    z = torch.randn(batch.size(0), 2)  # 随机噪声
    condition = torch.randn(batch.size(0), 32)  # 随机条件向量

    # 生成填补后的数据
    with torch.no_grad():
        imputed_data = generator(batch, z, condition)

    # 输出填补结果
    print("Imputed Data:")
    print(imputed_data.numpy())

```