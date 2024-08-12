import argparse
import torch


def get_configuration():
    parser = argparse.ArgumentParser(description='CondiGan for interpolation of time series data')

    # 训练参数
    parser.add_argument('-t', '--train', dest='train', type=bool, default=False, help='Whether to train a new model.')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=1826,
                        help='Specifies a random number seed. The default is 1826')
    parser.add_argument('-p', '--patience', dest='patience', type=int, default=5,
                        help='Set the patience parameter for early stop.')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=100,
                        help='Specify the number of training rounds.')
    parser.add_argument('-r', '--lr', dest='lr', type=float, default=0.0002, help='Assigned learning rate.')
    parser.add_argument('--train_size', dest='train_size', type=float, default=0.8, help='train_size.')

    # 配置数据集
    parser.add_argument('-tf', '--t-file', dest='t_file', type=str, default='./dataset/wind_0001_1h_train_4800.csv',
                        help='Specify the training data set')
    parser.add_argument('-if', '--i-file', dest='i_file', type=str, default='./dataset/wind_0001_1h_test_600.csv',
                        help='Specify the test data set')

    # 模型相关参数
    parser.add_argument('--input-dim', dest='input_dim', type=int, default=34,
                        help='Specifies the total input dimension. Manual setting is not recommended.')
    parser.add_argument('--features-dim', dest='features_dim', type=int, default=2,
                        help='Specifies the number of features in the data set to be used.')
    parser.add_argument('--cond-dim', dest='cond_dim', type=int, default=32,
                        help='Specifies the conditional vector dimension.')
    parser.add_argument('--cond-emb-dim', dest='cond_emb_dim', type=int, default=65,
                        help='Specifies the conditional vector embedding dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int, default=1, help='Specify output dimension.')
    parser.add_argument('--noise-dim', dest='noise_dim', type=int, default=4,
                        help='Specifies the random noise dimension.')
    parser.add_argument('--noise-emb-dim', dest='noise_emb_dim', type=int, default=65,
                        help='Specifies the random noise embedding dimension.')
    parser.add_argument('--seq-length', dest='seq_length', type=int, default=64, help='Specified sequence length.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='Specify batch size.')
    parser.add_argument('--num-layers', dest='num_layers', type=int, default=2,
                        help='Specify the number of Transfomer layers.')
    parser.add_argument('--num-heads', dest='num_heads', type=int, default=4,
                        help='Specifies the number of Transfomer heads.')
    parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=32,
                        help='Specifies the Transfomer latent spatial dimension.')
    parser.add_argument('--d-model', dest='d_models', type=int, default=132,
                        help='Specify the feature dimension for each input and output sequence element.')

    args = parser.parse_args()

    return args


def configuration_override(args):
    # 与模型名称相关的参数
    args.seed = 1826
    args.batch_size = 64
    args.g_lr = 0.0002
    args.d_lr = 0.0004
    args.epochs = 100

    # 训练相关参数
    args.patience = 5
    args.t_file = './dataset/1h/wind_0001_1h_10k.csv'
    args.i_file = './dataset/1h/wind_0001_1h_test_600.csv'
    args.train_size = 0.8
    args.column_names = ['windSpeed2m', 'windSpeed10m']
    args.missing_mode = 'continuous'
    args.missing_rate = 0.2
    args.max_missing_length = 48

    # 其他参数，训练时可以适当修改，与模型无关
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.hidden_size = 64  # 隐藏层大小
    args.num_layers = 4  # Transformer层数
    args.num_heads = 4  # Transformer头数
    args.latent_dim = 32  # 潜在空间维度
    args.output_dim = 1  # 输出维度
    args.seq_length = 64  # 序列长度

    # !!! 不建议修改，如修改需要充分考虑模型的变化
    args.cond_dim = 91  # 条件向量维度
    args.features_dim = len(args.column_names)  # 特征维度
    args.input_dim = args.cond_dim + args.features_dim  # 总输入维度 = cond_dim + features_dim = 91 + 2 = 93
    args.noise_dim = 4  # 随机噪声的维度
    args.cond_emb_wind_dim = 16
    args.cond_emb_hourly_dim = 36
    args.cond_emb_daily_dim = 16
    args.cond_emb_weekly_dim = 64
    args.cond_emb_dim = args.cond_emb_wind_dim + args.cond_emb_hourly_dim + \
                        args.cond_emb_daily_dim + args.cond_emb_weekly_dim      # 16+36+16+64=132

    args.noise_emb_dim = 17  # 随机噪声嵌入维度
    # 每个输入和输出序列元素的特征维度，确保被num_heads整除,
    # 确保 args.input_dim + condition_emb_dim + z_emb_dim = d_model
    args.d_model = args.input_dim + args.cond_emb_dim + args.noise_emb_dim + args.features_dim   # 93+132+17+2=244


if __name__ == '__main__':
    pass
