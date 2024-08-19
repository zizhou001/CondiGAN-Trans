from experiment.training import train
from experiment.test import interpolate
from utils.config import get_configuration, configuration_override


def main():
    # 从命令行获取参数
    args = get_configuration()

    # 根据需要重写部分参数
    configuration_override(args)

    # 设置模型文件的文件名
    tmp_str = 's{}_bs{}_hs{}_e{}_nl{}_sl{}_p{}_{}_mr{}'.format(args.seed, args.batch_size, args.hidden_size,
                                                               args.epochs, args.num_layers,
                                                               args.seq_length, args.patience,
                                                               args.missing_mode, args.missing_rate)
    g_l_str = 'l{}_'.format(args.g_lr)
    d_l_str = 'l{}_'.format(args.d_lr)
    generator_saved_name = "G_" + g_l_str + tmp_str
    discriminator_saved_name = "D_" + d_l_str + tmp_str

    # 训练模型
    generator, discriminator = train(args, generator_saved_name, discriminator_saved_name)

    # 测试或验证
    interpolate(generator, args, remark=tmp_str)


if __name__ == "__main__":
    main()
