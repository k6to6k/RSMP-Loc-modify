# RSMP-Loc++修改指南

本指南将帮助您实施RSMP-Loc++论文修改方案中的"上下文感知的动态原型"模块，并进行相关测试。

## 1. 修复当前代码问题

在运行测试之前，我们需要修复一个已知的代码问题：

```bash
# 运行修复脚本
python fix_model.py
```

这个脚本会备份原始的`model.py`文件，并修复PredictionModule方法中可能导致测试失败的变量未定义问题。

## 2. 验证动态原型模块功能

我们提供了两个测试脚本：

### 2.1 简化测试（推荐）

首先运行简化的测试脚本，专门验证动态原型模块的核心功能：

```bash
bash run_prototype_test.sh
```

或者直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore ./test_prototype.py
```

### 2.2 完整测试

如果简化测试通过，可以运行完整的测试脚本来验证所有功能：

```bash
bash run_tests.sh
```

或者直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore ./test_dynamic_prototype.py
```

## 3. 检查实现细节

RSMP-Loc++的上下文感知动态原型模块包含以下关键部分：

### 3.1 ContextPrototypeModulator类

这个类实现了上下文感知的原型调制功能，能够根据视频的上下文信息动态调整全局原型。核心实现包括：

- 特征融合：结合全局原型和上下文向量
- MLP网络：生成适合当前上下文的原型调整残差
- Tanh激活：限制残差范围，确保调整幅度适中

### 3.2 原型记忆库

借鉴HR-Pro的设计，我们实现了全局原型库并使用动量更新机制：

- `prototype_memory`：存储各类别的全局原型表示
- `update_prototype_memory`：根据类别标签动量更新原型
- `prototype_momentum`：控制更新速度的超参数（默认0.99）

### 3.3 上下文向量提取

从视频特征中提取上下文信息：

- `extract_context_vector`：通过时间维度的平均池化获取视频级上下文向量

### 3.4 动态原型预测

PredictionModule方法的增强版本：

- 动态/静态原型切换：通过`use_dynamic_prototype`参数控制
- 批量处理：支持对一个批次内的每个样本生成不同的动态原型
- 动态类别表示：每个视频拥有特定于其上下文的类别表示

## 4. 开始训练

修改已通过测试后，可以使用标准训练脚本开始训练：

```bash
bash train.sh
```

## 5. 注意事项

- 动态原型模块需要在第二阶段训练时激活
- 如果遇到任何问题，请检查日志并参考测试输出
- 上下文感知模块参数可在Model_GAI类初始化部分进行调整

## 6. 理论依据

本实现融合了两个SOTA方法的优势：

- 从HR-Pro借鉴了原型库的概念和动量更新机制
- 从SQL-Net借鉴了上下文感知和查询学习的思想

通过结合这两者的优势，我们的动态原型模块能够更好地适应不同视觉外观的同类动作，提高模型的泛化能力。

