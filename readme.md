# 版本更迭

## v1.0

### 版本特性

1. 提取**小时信息**和**风向信息**作为条件向量输入
2. 主要插补`windSpeed3s`和`windSpeed2m`两列的风速数据
3. 考虑了随机噪声
4. 使用`Transfomer`作为生成器

### 展望

期望在未来版本更新的功能：

- **多尺度特征提取**：使用不同时间尺度的输入数据（如小时、天、周）来生成多个条件向量，将这些条件结合到生成器中，以捕捉时间序列中的长期和短期依赖关系。
- **自适应条件权重**：在条件向量的处理上，引入自适应机制，根据生成过程中的状态动态调整条件权重。可以使用小型神经网络来计算条件的权重，增强特定条件对生成结果的影响。
- **改进对抗训练**：在训练过程中，不仅使用判别器的输出作为反馈，还可以引入其他指标（如生成数据的统计特性）进行对抗训练，促使生成器产生更符合实际分布的数据。
- **动态输入序列长度**：考虑实现一个机制，根据输入数据的缺失程度动态调整生成器的输入序列长度，这样生成器可以适应不同长度的时间序列数据。

## v2.0

### 版本特性

1. 加入多尺度特征提取功能，模型能更好的捕捉时间序列中的长期和短期依赖关系


# 代码优化
## Stop too Early

### 问题描述

模型在训练的早期就触发了早停，这可能是因为以下几个原因：

1. **学习率过高**：过高的学习率可能导致损失震荡，而无法有效下降。尝试降低学习率。

2. **模型容量**：模型可能过于复杂或过于简单，导致学习不稳定。你可以考虑调整模型的层数或节点数。

3. **早停耐心值**：可以考虑将耐心值增加到15-20个epoch，以允许模型有更多时间进行学习。

4. **数据预处理**：确保数据预处理（如归一化、标准化）是合理的，这可能影响模型的收敛速度。

5. **批量大小**：调整批量大小（batch size），较小的批量可以提供更多的更新频率，但会导致更大的噪声。

### 优化建议：
- **降低学习率**：尝试将学习率减少一半，观察验证损失是否稳定下降。
- **增加耐心值**：调整耐心值，给予模型更多学习机会。
- **检查数据预处理**：确保输入数据已适当处理。
- **实验不同的模型结构**：考虑简化或复杂化模型，查看对训练和验证损失的影响。

通过以上步骤，你可能能够改善模型的学习过程，减少早停的发生。

