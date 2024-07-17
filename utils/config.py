import argparse


def get_configuration():
    parser = argparse.ArgumentParser(description='CondiGan for interpolation of time series data')

    parser.add_argument('-t', '--train', dest='train', type=bool, default=False, help='Whether to train a new model.')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=1826,
                        help='Specifies a random number seed. The default is 1826')
    parser.add_argument('-f', '--file-path', dest='file_path', type=str, default='./dataset/wind_0001_1h.csv',
                        help='Specifies a random number seed. The default is 1826')
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
    parser.add_argument('--num-heads', dest='num_heads', type=int, default=4,
                        help='Specifies the number of Transfomer heads.')

    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=100,
                        help='Specify the number of training rounds.')
    parser.add_argument('-r', '--lr', dest='lr', type=float, default=0.0002, help='Assigned learning rate.')

    args = parser.parse_args()

    return args


def configuration_override(args):

    # 与模型名称相关的参数
    args.seed = 1826
    args.batch_size = 32
    args.lr = 0.0002
    args.epochs = 100

    args.file_path = './dataset/wind_0001_1h.csv'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.hidden_size = 64  # 隐藏层大小
    args.num_layers = 2  # Transformer层数
    args.num_heads = 4  # Transformer头数
    args.latent_dim = 32  # 潜在空间维度

    args.input_dim = 34  # 总输入维度
    args.features_dim = 2  # 特征维度
    args.cond_dim = 32  # 条件向量维度
    args.cond_emb_dim = 65  # 条件向量嵌入维度
    args.output_dim = 1  # 输出维度
    args.noise_dim = 4  # 随机噪声的维度
    args.noise_emb_dim = 65  # 随机噪声嵌入维度
    args.seq_length = 64  # 序列长度
    # 每个输入和输出序列元素的特征维度，确保被num_heads整除,
    # 确保 condition_emb_dim + feature + z_emb_dim = d_model
    args.d_model = 132




if __name__ == '__main__':
    pass
