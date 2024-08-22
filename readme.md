# 使用说明

## 训练环境

本环境基于pytorch搭建，可使用下面的配置文件创建相应的虚拟环境

```yaml
name: msc-gan
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/fastai/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
dependencies:
  - blas=1.0=mkl
  - bottleneck=1.3.7=py39h9128911_0
  - brotli=1.0.9=h2bbff1b_8
  - brotli-bin=1.0.9=h2bbff1b_8
  - ca-certificates=2024.7.2=haa95532_0
  - contourpy=1.2.0=py39h59b6b97_0
  - cycler=0.11.0=pyhd3eb1b0_0
  - fonttools=4.51.0=py39h2bbff1b_0
  - freetype=2.12.1=ha860e81_0
  - icc_rt=2022.1.0=h6049295_2
  - icu=73.1=h6c2663c_0
  - importlib_resources=6.4.0=py39haa95532_0
  - intel-openmp=2023.1.0=h59b6b97_46320
  - joblib=1.4.2=py39haa95532_0
  - jpeg=9e=h827c3e9_3
  - kiwisolver=1.4.4=py39hd77b12b_0
  - krb5=1.20.1=h5b6d351_0
  - lcms2=2.12=h83e58a3_0
  - lerc=3.0=hd77b12b_0
  - libbrotlicommon=1.0.9=h2bbff1b_8
  - libbrotlidec=1.0.9=h2bbff1b_8
  - libbrotlienc=1.0.9=h2bbff1b_8
  - libclang=14.0.6=default_hb5a9fac_1
  - libclang13=14.0.6=default_h8e68704_1
  - libdeflate=1.17=h2bbff1b_1
  - libpng=1.6.39=h8cc25b3_0
  - libpq=12.17=h906ac69_0
  - libtiff=4.5.1=hd77b12b_0
  - libwebp-base=1.3.2=h2bbff1b_0
  - lz4-c=1.9.4=h2bbff1b_1
  - matplotlib=3.8.4=py39haa95532_0
  - matplotlib-base=3.8.4=py39h4ed8f06_0
  - mkl=2023.1.0=h6b88ed4_46358
  - mkl-service=2.4.0=py39h2bbff1b_1
  - mkl_fft=1.3.8=py39h2bbff1b_0
  - mkl_random=1.2.4=py39h59b6b97_0
  - numexpr=2.8.7=py39h2cd9be0_0
  - numpy-base=1.26.4=py39h65a83cf_0
  - openjpeg=2.5.2=hae555c5_0
  - openssl=3.0.14=h827c3e9_0
  - packaging=24.1=py39haa95532_0
  - pandas=2.2.2=py39h5da7b33_0
  - pillow=10.4.0=py39h827c3e9_0
  - pip=24.2=py39haa95532_0
  - ply=3.11=py39haa95532_0
  - pybind11-abi=5=hd3eb1b0_0
  - pyparsing=3.0.9=py39haa95532_0
  - pyqt=5.15.10=py39hd77b12b_0
  - pyqt5-sip=12.13.0=py39h2bbff1b_0
  - python=3.9.19=h1aa4202_1
  - python-dateutil=2.9.0post0=py39haa95532_2
  - python-tzdata=2023.3=pyhd3eb1b0_0
  - pytz=2024.1=py39haa95532_0
  - qt-main=5.15.2=h19c9488_10
  - scikit-learn=1.5.1=py39hc64d2fc_0
  - scipy=1.13.1=py39h8640f81_0
  - setuptools=72.1.0=py39haa95532_0
  - sip=6.7.12=py39hd77b12b_0
  - six=1.16.0=pyhd3eb1b0_1
  - sqlite=3.45.3=h2bbff1b_0
  - tbb=2021.8.0=h59b6b97_0
  - threadpoolctl=3.5.0=py39h9909e9c_0
  - tomli=2.0.1=py39haa95532_0
  - tornado=6.4.1=py39h827c3e9_0
  - tzdata=2024a=h04d1e81_0
  - unicodedata2=15.1.0=py39h2bbff1b_0
  - vc=14.40=h2eaa2aa_0
  - vs2015_runtime=14.40.33807=h98bb1dd_0
  - wheel=0.43.0=py39haa95532_0
  - xz=5.4.6=h8cc25b3_1
  - zipp=3.17.0=py39haa95532_0
  - zlib=1.2.13=h8cc25b3_1
  - zstd=1.5.5=hd43e919_2
  - pip:
    - certifi==2024.7.4
    - charset-normalizer==3.3.2
    - idna==3.7
    - numpy==2.0.1
    - requests==2.32.3
    - torch==1.13.1+cu116
    - torchaudio==0.13.1+cu116
    - torchvision==0.14.1+cu116
    - typing-extensions==4.12.2
    - urllib3==2.2.2
prefix: F:\anaconda\envs\msc-gan
```


# 版本更迭

## v1.0

### 版本特性

1. 提取**小时信息**和**风向信息**作为条件向量输入
2. 主要插补`windSpeed3s`和`windSpeed2m`两列的风速数据
3. 考虑了随机噪声
4. 使用`Transfomer`作为生成器

### bug

- 未测试模型效果
- 未编写测试集上的逻辑

### 展望

期望在未来版本更新的功能：

- **多尺度特征提取**：使用不同时间尺度的输入数据（如小时、天、周）来生成多个条件向量，将这些条件结合到生成器中，以捕捉时间序列中的长期和短期依赖关系。
- **自适应条件权重**：在条件向量的处理上，引入自适应机制，根据生成过程中的状态动态调整条件权重。可以使用小型神经网络来计算条件的权重，增强特定条件对生成结果的影响。
- **改进对抗训练**：在训练过程中，不仅使用判别器的输出作为反馈，还可以引入其他指标（如生成数据的统计特性）进行对抗训练，促使生成器产生更符合实际分布的数据。
- **动态输入序列长度**：考虑实现一个机制，根据输入数据的缺失程度动态调整生成器的输入序列长度，这样生成器可以适应不同长度的时间序列数据。

## v2.0

### 版本特性

1. 加入多尺度特征提取功能，模型能更好的捕捉时间序列中的长期和短期依赖关系

### bug

- 未测试模型效果
- 未编写测试集上的逻辑

## v2.1

### 版本特性

1. 修改普通判别器为多尺度判别器以对应多尺度输入。

### attention

- 模型主要实现的功能是预测，而不是插补


## v2.2

### 版本特性

1. 实现插补的功能，测试通过

## v2.3

### 版本特性

1. 优化了绘图函数
2. 增加了数据集
3. 修改生成器bug
4. 优化训练策略 
5. 编写了相应的脚本，用于训练模型
