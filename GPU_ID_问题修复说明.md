# GPU_ID 参数设置问题修复说明

## 问题描述

在项目中发现了两个与 GPU 设备相关的问题：

1. 在 `train.sh` 中设置 `GPU_ID=2`，但实际训练时仍然使用 1 号 GPU，导致无法按预期切换 GPU。

2. 在 `eval.sh` 中运行评估脚本时，出现模型加载错误：
   ```
   RuntimeError: Attempting to deserialize object on CUDA device 2 but torch.cuda.device_count() is 2.
   ```

## 问题原因

经过分析，发现了以下几个问题：

1. **训练脚本问题**：
   - **device 变量引用错误**：在 `train.py` 文件中使用了 `device` 变量，但该变量并未在函数中定义，应该使用 `config.device`。
   - **config 变量未定义**：在 `Total_loss` 和 `Total_loss_Gai` 类的 `forward` 方法中使用了 `config` 变量，但该变量并未作为参数传入。

2. **评估脚本问题**：
   - **模型加载缺少 map_location 参数**：在 `main_eval.py` 中加载模型时没有指定 `map_location` 参数，导致无法将模型从一个 GPU 设备映射到另一个设备。

## 解决方案

我们通过以下步骤解决了这些问题：

### 训练脚本修复

1. **修复 device 引用**：将 `train.py` 中所有的 `.to(device)` 改为 `.to(config.device)`。

2. **添加 config 参数**：修改 `Total_loss` 和 `Total_loss_Gai` 类的 `forward` 方法，添加一个可选的 `config` 参数。

3. **传递 config 参数**：修改 `train` 函数中的 `criterion` 调用，传递 `config` 参数。

### 评估脚本修复

1. **添加 map_location 参数**：修改 `main_eval.py` 中的模型加载代码，添加 `map_location` 参数：
   ```python
   net.load_state_dict(torch.load(args.model_file, map_location=config.device))
   ```

## 修复文件

1. **fix_train.py**：修复 `train.py` 中的 `device` 引用问题。
2. **fix_config_param.py**：修复 `train.py` 中的 `config` 变量未定义问题。
3. **fix_eval.py**：修复 `main_eval.py` 中的模型加载问题。

## 使用方法

按照以下步骤应用修复：

1. 运行 `python fix_train.py` 修复 device 引用问题
2. 运行 `python fix_config_param.py` 修复 config 变量未定义问题
3. 运行 `python fix_eval.py` 修复评估脚本中的模型加载问题
4. 现在可以通过修改 `train.sh` 和 `eval.sh` 中的 `GPU_ID` 参数来选择使用的 GPU

## 验证方法

可以通过以下命令验证 GPU 设置是否生效：

```bash
python check_gpu.py --gpu_id 2
```

输出应显示：
```
命令行参数中的GPU_ID: 2
使用的设备: cuda:2
CUDA是否可用: True
...
```

然后运行训练和评估脚本：
```bash
bash train.sh
bash eval.sh
```

训练和评估应该在指定的 GPU 上进行。