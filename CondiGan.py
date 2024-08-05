import torch
import torch.nn as nn
from utils.dataset import multiscale_divider


class Generator(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, input_dim, seq_length,
                 cond_dim, noise_dim, noise_emb_dim, cond_emb_wind_dim,
                 features_dim, cond_emb_hourly_dim, cond_emb_daily_dim,
                 cond_emb_weekly_dim):
        super(Generator, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.cond_dim = cond_dim
        self.z_dim = noise_dim
        self.z_emb_dim = noise_emb_dim
        self.cond_emb_wind_dim = cond_emb_wind_dim
        self.features_dim = features_dim
        self.cond_emb_wind_dim = cond_emb_wind_dim
        self.cond_emb_hourly_dim = cond_emb_hourly_dim
        self.cond_emb_daily_dim = cond_emb_daily_dim
        self.cond_emb_weekly_dim = cond_emb_weekly_dim

        # 条件向量嵌入层
        self.condition_embedding_wind = nn.Linear(8, self.cond_emb_wind_dim)
        self.condition_embedding_hourly = nn.Linear(24, self.cond_emb_wind_dim)
        self.condition_embedding_daily = nn.Linear(7, self.cond_emb_hourly_dim)
        self.condition_embedding_weekly = nn.Linear(52, self.cond_emb_weekly_dim)

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
        condition_emb_hourly = self.condition_embedding_hourly(hourly_condition.float())
        condition_emb_daily = self.condition_embedding_daily(daily_condition.float())
        condition_emb_weekly = self.condition_embedding_weekly(weekly_condition.float())
        condition_emb_wind = self.condition_embedding_wind(wind_condition.float())

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

        # 合并 (batch_size, seq_length, input_dim + 4 * 条件向量嵌入 + noise_emb_dim)
        x_with_condition_z = torch.cat(
            (initial_input, condition_emb_hourly, condition_emb_daily, condition_emb_weekly, condition_emb_wind, z_emb),
            dim=-1)

        # 转换为 Transformer 输入格式，调整张量的维度顺序
        x_with_condition_z = x_with_condition_z.permute(1, 0,
                                                        2)  # (seq_length, batch_size, input_dim + 4 * cond_emb_dim + noise_emb_dim)

        # 通过 Transformer 编码器
        x_transformed = self.transformer(x_with_condition_z)

        # 转换回原始形状
        x_transformed = x_transformed.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        # 输出处理
        x_output = self.linear(x_transformed)  # (batch_size, seq_length, features_dim)

        return x_output


# 多尺度判别器
class Discriminator(nn.Module):
    def __init__(self, features_dim=2, cond_dim=32, hidden_size=64):
        super(Discriminator, self).__init__()

        self.features_dim = features_dim
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size

        # 定义多尺度分支
        self.hourly_branch = self.build_branch(24)
        self.daily_branch = self.build_branch(7)
        self.weekly_branch = self.build_branch(52)
        self.wind_branch = self.build_branch(8)

        # 最终合并层
        self.combine_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),  # 4个分支的合并
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def build_branch(self, dim):
        return nn.Sequential(
            nn.Linear(self.features_dim + dim, self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, condition):
        hourly_condition, daily_condition, weekly_condition, wind_condition = multiscale_divider(condition)

        # 准备每个分支的输入
        inputs_hourly = torch.cat([x, hourly_condition.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)
        inputs_daily = torch.cat([x, daily_condition.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)
        inputs_weekly = torch.cat([x, weekly_condition.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)
        inputs_wind = torch.cat([x, wind_condition.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)

        # 分别通过每个分支
        output_hourly = self.hourly_branch(inputs_hourly.view(-1, inputs_hourly.size(-1)))
        output_daily = self.daily_branch(inputs_daily.view(-1, inputs_daily.size(-1)))
        output_weekly = self.weekly_branch(inputs_weekly.view(-1, inputs_weekly.size(-1)))
        output_wind = self.wind_branch(inputs_wind.view(-1, inputs_wind.size(-1)))

        # 合并所有分支的输出
        combined_output = torch.cat([output_hourly, output_daily, output_weekly, output_wind], dim=-1)
        final_output = self.combine_layer(combined_output)

        # 重塑以匹配输入 x 的尺寸
        final_output = final_output.view(x.size(0), x.size(1), -1)

        return final_output
