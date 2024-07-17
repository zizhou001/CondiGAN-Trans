import torch
import torch.nn as nn


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

    def forward(self, x, z, condition):
        """
        :param x: 从数据集中提取的风速数据，形状应为 (batch_size, seq_length, 2)
        :param z: 随机噪声
        :param condition: 条件向量，形状应为 (batch_size, cond_dim)
        :return: 生成的风速数据，形状为 (batch_size, seq_length, input_dim)
        """

        # 打印 x 和 condition 的形状以进行调试
        """
        print(f"[x] G > forward : {x.shape}")  # 应该是 (batch_size, seq_length, 2)
        print(f"[condition] G > forward : {condition.shape}")  # 应该是 (batch_size, cond_dim)
        """

        # 确保 x 的维度为 (batch_size, seq_length, 2)
        if x.dim() != 3 or x.size(1) != self.seq_length or x.size(2) != 2:
            raise ValueError(f"Unexpected x shape: {x.shape}")

        # 生成条件嵌入
        condition_emb = self.condition_embedding(condition)  # (batch_size, d_model - input_dim - z_dim)

        # 噪声嵌入
        z_emb = self.z_dim(z)  # (batch_size, d_model - input_dim)

        # 检查条件嵌入之前的形状
        """
        print(f"[x] G > forward > emb: {x.shape}")  # 应该是 (batch_size, seq_length, features)
        print(f"[condition_emb] G > forward > emb: {condition_emb.shape}")  # (batch_size, d_model - input_dim - z_dim)
        print(f"[z_emb] G > forward > BEFORE > emb: {z_emb.shape}")  # (batch_size, d_model - input_dim)
        """

        # 扩展嵌入
        condition_emb = condition_emb.unsqueeze(1).repeat(1, self.seq_length, 1)  # (batch_size, seq_length, d_model)
        z_emb = z_emb.unsqueeze(1).repeat(1, self.seq_length, 1)  # (batch_size, seq_length, z_emb_dim)

        # 检查条件嵌入的形状
        """
        print(f"[condition_emb] G > forward > unsqueeze: {condition_emb.shape}") 
        print(f"[z_emb] G > forward > unsqueeze: {z_emb.shape}")  
        print(f"[x] G > forward > unsqueeze: {x.shape}")  
        """

        # 合并 (batch_size, seq_length, feature + cond_emb_dim + cond_emb_dim)
        x_with_condition_z = torch.cat((x, condition_emb, z_emb), dim=-1)


        # 转换为 Transformer 输入格式，调整张量的维度顺序
        x_with_condition_z = x_with_condition_z.permute(1, 0, 2)  # 转换为 (seq_length, batch_size, features)
        # print(x_with_condition_z.shape)

        # 通过 Transformer 编码器
        x_transformed = self.transformer(x_with_condition_z)

        # 转换回原始形状
        x_transformed = x_transformed.permute(1, 0, 2)  # 转换回 (batch_size, seq_length, d_model)

        # 输出处理
        x_output = self.linear(x_transformed)  # (batch_size, seq_length, input_dim)

        # print(f"[x_output] G > forward > return: ", {x_output.shape})
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
        :param condition: 条件信息，形状为 (batch_size, cond_dim)
        :return: 经过判别器处理后的输出，形状为 (batch_size, seq_length, 1)
        """

        """
        print(f"[x] D > forward : {x.shape}")  # 应该是 (batch_size, seq_length, input_dim)
        print(f"[condition] D > forward: {condition.shape}")  # (batch_size, cond_dim)
        """

        # 重复条件信息以匹配序列长度
        condition = condition.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, seq_length, cond_dim)

        # 合并数据和条件信息
        x_with_condition = torch.cat([x, condition], dim=-1)  # 形状为 (batch_size, seq_length, input_dim + cond_dim)

        # 通过判别器模型
        x_out = self.model(x_with_condition.view(-1, x_with_condition.size(-1)))  # 展平输入
        x_out = x_out.view(x.size(0), x.size(1), -1)  # 恢复形状为 (batch_size, seq_length, 1)

        return x_out
