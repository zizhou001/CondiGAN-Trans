import argparse
import torch


def get_configuration():
    parser = argparse.ArgumentParser(description='CondiGan for interpolation of time series data')

    # 训练参数
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64, help='Specify batch size.')
    parser.add_argument('--num-layers', dest='num_layers', type=int, default=6,
                        help='Specify the number of Transfomer layers.')
    parser.add_argument('--patience', dest='patience', type=int, default=10,
                        help='Set the patience parameter for early stop.')

    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=64, help='Specify hidden_size.')
    parser.add_argument('--seq-length', dest='seq_length', type=int, default=64, help='Specified sequence length.')
    parser.add_argument('--missing-rate', dest='missing_rate', type=float, default=0.2, help='Set missing_rate.')
    parser.add_argument('--max-missing-rate', dest='max_missing_rate', type=float, default=0.3,
                        help='Set max_missing_rate.')

    args = parser.parse_args()
    args = more_settings(args, model_size="default")

    return args


def configuration_override(args):
    # 训练相关参数
    args.seed = 1826
    args.batch_size = 64
    args.g_lr = 0.0001
    args.d_lr = 0.0002
    args.epochs = 100
    args.patience = 5
    args.file_path = './dataset/1h/wind_0001_1h_11k.csv'
    args.train_size = 0.8
    args.column_names = ['windSpeed2m', 'windSpeed10m']
    args.missing_mode = 'continuous'
    args.missing_rate = 0.9
    args.max_missing_rate = 0.7
    args.max_missing_length = 600 * args.missing_rate * args.max_missing_rate
    args.num_layers = 6  # Transformer层数
    args.seq_length = 128  # 序列长度(64)
    args.hidden_size = 64  # 隐藏层大小(64)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 其他参数
    args.latent_dim = 32  # 潜在空间维度
    args.output_dim = 1  # 输出维度

    # !!! 不建议修改，如修改需要充分考虑模型的变化
    args.num_heads = 4  # Transformer头数
    args.cond_dim = 91  # 条件向量维度
    args.features_dim = len(args.column_names)  # 特征维度
    args.input_dim = args.cond_dim + args.features_dim  # 总输入维度 = cond_dim + features_dim = 91 + 2 = 93
    args.noise_dim = 4  # 随机噪声的维度
    args.cond_emb_wind_dim = 16
    args.cond_emb_hourly_dim = 36
    args.cond_emb_daily_dim = 16
    args.cond_emb_weekly_dim = 64
    args.cond_emb_dim = args.cond_emb_wind_dim + args.cond_emb_hourly_dim + \
                        args.cond_emb_daily_dim + args.cond_emb_weekly_dim  # 16+36+16+64=132

    args.noise_emb_dim = 17  # 随机噪声嵌入维度
    # 每个输入和输出序列元素的特征维度，确保被num_heads整除,
    # 确保 args.input_dim + condition_emb_dim + z_emb_dim = d_model
    args.d_model = args.input_dim + args.cond_emb_dim + args.noise_emb_dim + args.features_dim  # 93+132+17+2=244


def more_settings(args, model_size="default"):
    # 训练相关参数
    args.seed = 1826
    args.g_lr = 0.0001
    args.d_lr = 0.0002
    args.epochs = 100
    args.file_path = './dataset/1h/wind_0001_1h_11k.csv'
    args.column_names = ['windSpeed2m', 'windSpeed10m']
    args.missing_mode = 'continuous'
    args.max_missing_length = 600 * args.missing_rate * args.max_missing_rate
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.latent_dim = 32
    args.output_dim = 1

    if model_size == 'large':
        args.num_heads = 4  # Transformer头数
        args.cond_dim = 91  # 条件向量维度
        args.features_dim = len(args.column_names)  # 特征维度
        args.input_dim = args.cond_dim + args.features_dim  # 总输入维度 = cond_dim + features_dim = 91 + 2 = 93
        args.noise_dim = 16  # 随机噪声的维度

        # 根据比例分配条件嵌入维度
        args.cond_emb_wind_dim = 36
        args.cond_emb_hourly_dim = 105
        args.cond_emb_daily_dim = 30
        args.cond_emb_weekly_dim = 228

        args.cond_emb_dim = args.cond_emb_wind_dim + args.cond_emb_hourly_dim + \
                            args.cond_emb_daily_dim + args.cond_emb_weekly_dim  # 36 + 105 + 30 + 228 = 399

        args.noise_emb_dim = 16  # 调整后的随机噪声嵌入维度

        args.d_model = args.input_dim + args.cond_emb_dim + args.noise_emb_dim + args.features_dim  # 93 + 399 + 16 + 2 = 510
        return args

    elif model_size == 'default':
        args.num_heads = 4  # Transformer头数
        args.cond_dim = 91  # 条件向量维度
        args.features_dim = len(args.column_names)  # 特征维度
        args.input_dim = args.cond_dim + args.features_dim  # 总输入维度 = cond_dim + features_dim = 91 + 2 = 93
        args.noise_dim = 4  # 随机噪声的维度

        # 根据比例分配条件嵌入维度
        total_ratio = 8 + 24 + 7 + 52  # 总比例
        args.cond_emb_wind_dim = round(145 * (8 / total_ratio))  # 13
        args.cond_emb_hourly_dim = round(145 * (24 / total_ratio))  # 38
        args.cond_emb_daily_dim = round(145 * (7 / total_ratio))  # 11
        args.cond_emb_weekly_dim = round(145 * (52 / total_ratio))  # 83

        args.cond_emb_dim = args.cond_emb_wind_dim + args.cond_emb_hourly_dim + \
                            args.cond_emb_daily_dim + args.cond_emb_weekly_dim  # 13 + 38 + 11 + 83 = 145

        args.noise_emb_dim = 16  # 调整后的随机噪声嵌入维度

        args.d_model = args.input_dim + args.cond_emb_dim + args.noise_emb_dim + args.features_dim  # 93 + 145 + 16 + 2 = 254
        return args

    return args


if __name__ == '__main__':
    pass
