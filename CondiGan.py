import torch
import torch.nn as nn
from utils.dataset import multiscale_divider


class Generator(nn.Module):
    def __init__(self, d_model=132,
                 num_heads=4,
                 num_layers=2,
                 input_dim=34,
                 seq_length=64,
                 cond_dim=32,
                 noise_dim=2,
                 cond_emb_dim=64,
                 noise_emb_dim=64,
                 features_dim=2):
        super(Generator, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.cond_dim = cond_dim
        self.z_dim = noise_dim
        self.z_emb_dim = noise_emb_dim
        self.cond_emb_dim = cond_emb_dim
        self.features_dim = features_dim

        # 条件向量嵌入
        self.condition_embedding = nn.Linear(self.cond_dim, self.cond_emb_dim)

        # 随机噪声嵌入
        self.z_dim = nn.Linear(self.z_dim, self.z_emb_dim)  # 随机噪声嵌入

        # transformer层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads),
            num_layers=self.num_layers
        )

        # 输出层
        self.linear = nn.Linear(self.d_model, self.features_dim)  # 根据输入维度调整线性层

    def forward(self, z, condition):
        """
        :param z: 随机噪声，形状为 (batch_size, noise_dim)
        :param condition: 条件向量 (batch_size, cond_dim)
            hourly_condition: 小时尺度条件向量，形状为 (batch_size, 24)
            daily_condition: 日尺度条件向量，形状为 (batch_size, 7)
            weekly_condition: 周尺度条件向量，形状为 (batch_size, 52)
            wind_condition: 风向条件向量，形状为 (batch_size, 8)
        :return: 生成的风速数据，形状为 (batch_size, seq_length, input_dim)
        """

        # 获取多尺度数据
        hourly_condition, daily_condition, weekly_condition, wind_condition = multiscale_divider(condition)

        # 生成条件嵌入
        condition_emb_hourly = self.condition_embedding(hourly_condition.float())  # 形状为 (batch_size, cond_emb_dim)
        condition_emb_daily = self.condition_embedding(daily_condition.float())  # 形状为 (batch_size, cond_emb_dim)
        condition_emb_weekly = self.condition_embedding(weekly_condition.float())  # 形状为 (batch_size, cond_emb_dim)
        condition_emb_wind = self.condition_embedding(wind_condition.float())  # 形状为 (batch_size, cond_emb_dim)

        # 噪声嵌入
        z_emb = self.z_dim(z)  # (batch_size, z_emb_dim)

        # 扩展嵌入
        condition_emb_hourly = condition_emb_hourly.unsqueeze(1).repeat(1, self.seq_length,
                                                                        1)  # (batch_size, seq_length, cond_emb_dim)
        condition_emb_daily = condition_emb_daily.unsqueeze(1).repeat(1, self.seq_length,
                                                                      1)  # (batch_size, seq_length, cond_emb_dim)
        condition_emb_weekly = condition_emb_weekly.unsqueeze(1).repeat(1, self.seq_length,
                                                                        1)  # (batch_size, seq_length, cond_emb_dim)
        condition_emb_wind = condition_emb_wind.unsqueeze(1).repeat(1, self.seq_length,
                                                                    1)  # (batch_size, seq_length, cond_emb_dim)
        z_emb = z_emb.unsqueeze(1).repeat(1, self.seq_length, 1)  # (batch_size, seq_length, z_emb_dim)

        # 创建初始输入
        initial_input = torch.zeros(z_emb.size(0), self.seq_length, self.input_dim).to(
            z_emb.device)  # (batch_size, seq_length, input_dim)

        # 合并条件向量和噪声嵌入
        x_with_conditions_z = torch.cat(
            (initial_input, condition_emb_hourly, condition_emb_daily, condition_emb_weekly, condition_emb_wind, z_emb),
            dim=-1)  # (batch_size, seq_length, input_dim + 4 * cond_emb_dim + z_emb_dim)

        # 转换为 Transformer 输入格式
        x_with_conditions_z = x_with_conditions_z.permute(1, 0,
                                                          2)  # (seq_length, batch_size, input_dim + 4 * cond_emb_dim + z_emb_dim)

        # 通过 Transformer 编码器
        x_transformed = self.transformer(x_with_conditions_z)

        # 转换回原始形状
        x_transformed = x_transformed.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        # 输出处理
        x_output = self.linear(x_transformed)  # (batch_size, seq_length, input_dim)

        return x_output


# 判别器模型
class Discriminator(nn.Module):
    def __init__(self, features_dim=2, cond_dim=32, hidden_size=64):
        super(Discriminator, self).__init__()

        self.features_dim = features_dim
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size

        # 输入维度 = 风速特征维度 + 条件嵌入维度
        self.model = nn.Sequential(
            nn.Linear(self.features_dim + self.cond_dim, self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # 添加Dropout以防止过拟合
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x, condition):
        """
        :param x: 输入的数据，形状为 (batch_size, seq_length, input_dim)
        :param condition: 条件向量 (batch_size, cond_dim)
            hourly_condition: 小时尺度条件向量，形状为 (batch_size, 24)
            daily_condition: 日尺度条件向量，形状为 (batch_size, 7)
            weekly_condition: 周尺度条件向量，形状为 (batch_size, 52)
            wind_condition: 风向条件向量，形状为 (batch_size, 8)
        :return: 经过判别器处理后的输出，形状为 (batch_size, seq_length, 1)
        """

        hourly_condition, daily_condition, weekly_condition, wind_condition = multiscale_divider(condition)

        # 重复条件信息以匹配序列长度
        condition_hourly = hourly_condition.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, seq_length, 24)
        condition_daily = daily_condition.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, seq_length, 7)
        condition_weekly = weekly_condition.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, seq_length, 52)
        wind_condition = wind_condition.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, seq_length, 8)

        # 合并数据和条件信息
        x_with_condition = torch.cat([x, condition_hourly, condition_daily, condition_weekly, wind_condition], dim=-1)
        # 形状为 (batch_size, seq_length, input_dim + cond_dim)

        # 通过判别器模型
        x_out = self.model(x_with_condition.view(-1, x_with_condition.size(-1)))  # 展平输入
        x_out = x_out.view(x.size(0), x.size(1), -1)  # 恢复形状为 (batch_size, seq_length, 1)

        return x_out
