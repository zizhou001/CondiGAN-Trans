from experiment.training import train
from experiment.test import interpolate
from utils.config import get_configuration, configuration_override


def main():
    # 从命令行获取参数
    args = get_configuration()

    # 根据需要重写部分参数
    configuration_override(args)

    # 设置模型文件的文件名
    tmp_str = 's{}_b{}_l{}_e{}'.format(args.seed, args.batch_size, args.lr, args.epochs)
    generator_saved_name = "G_" + tmp_str
    discriminator_saved_name = "D_" + tmp_str

    # 训练模型
    generator, discriminator = train(args, generator_saved_name, discriminator_saved_name)

    # 测试或验证
    missing_data_file = ""
    interpolate(generator, missing_data_file, args)



if __name__ == "__main__":
    main()
