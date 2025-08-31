# GPU_ID修改一步到位指南

## 1. 背景介绍

在深度学习项目中，特别是多GPU服务器环境下，正确设置和切换GPU设备对资源利用至关重要。本文档提供一步到位的解决方案，解决RSMP项目中GPU设备切换的所有问题。

## 2. 问题概述

项目中存在的GPU设备相关问题：

1. **训练脚本问题**：设置`GPU_ID=2`，但实际使用的仍是1号GPU
2. **评估脚本问题**：运行评估时出现模型加载错误：
   ```
   RuntimeError: Attempting to deserialize object on CUDA device 2 but torch.cuda.device_count() is 2.
   ```
3. **代码中的具体问题**：
   - 变量引用错误：使用`device`而非`config.device`
   - 未定义变量：损失函数中使用了未传入的`config`
   - 硬编码设备：代码中直接使用`cuda:1`等固定设备
   - 模型加载缺少`map_location`参数

## 3. 一步到位解决方案

### 3.1 确保配置文件正确

修改`config.py`文件，确保其正确设置GPU设备：

```python
# 在config.py中添加或修改以下内容
def parse_gpu_id(args):
    """根据命令行参数设置device"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cpu")

# 在配置类的初始化方法中
def __init__(self, args):
    # 其他配置代码...
    self.device = parse_gpu_id(args)
    # 其他配置代码...
```

### 3.2 修改命令行参数解析

修改`options.py`文件，确保正确接收GPU_ID参数：

```python
# 在options.py的参数解析器中添加
parser.add_argument('--gpu_id', type=int, default=1, help='使用的GPU设备ID (0, 1, 2, ...)')
```

### 3.3 修改训练脚本(`train.py`)

对`train.py`进行以下修改：

1. 替换所有的`.to(device)`为`.to(config.device)`
2. 为损失函数的`forward`方法添加`config`参数：
   ```python
   # 修改Total_loss和Total_loss_Gai类的forward方法
   def forward(self, vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, 
               label, point_anno, step, pseudo_instance_label, uncertainty, config=None):
       # 函数内容...
   ```
3. 在调用损失函数时传递`config`参数：
   ```python
   cost, loss = criterion(vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, 
                         _label, _point_anno, step, pseudo_instance_label=pseudo_instance_label, 
                         uncertainty=uncertainty, config=config)
   ```
4. 替换所有硬编码的GPU设备引用：
   ```python
   # 将所有类似这样的代码
   torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(device)
   # 修改为
   torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(config.device)
   
   # 将所有类似这样的代码
   torch.device('cuda:1')
   # 修改为
   config.device
   ```

### 3.4 修改评估脚本(`main_eval.py`)

在`main_eval.py`中添加模型加载时的设备映射参数：

```python
# 将
net.load_state_dict(torch.load(args.model_file))
# 修改为
net.load_state_dict(torch.load(args.model_file, map_location=config.device))
```

### 3.5 修改其他所有Python文件

使用以下步骤批量修改所有Python文件中的硬编码设备引用：

1. 搜索并替换所有`torch.device('cuda:1')`或`torch.device("cuda:1")`为`config.device`
2. 搜索并替换所有`.to('cuda:1')`或`.to("cuda:1")`为`.to(config.device)`
3. 检查可能存在的其他硬编码设备引用，如`'cuda:1'`作为参数传递的地方

### 3.6 修改训练和评估脚本文件

修改`train.sh`和`eval.sh`文件：

```bash
#!/bin/bash

# 设置要使用的GPU ID
GPU_ID=2  # 可以根据需要修改为其他值

# 设置可见的GPU设备（全部可见）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --num_iters 200 \
    --epochs_per_step 50 \
    --batch_size 16 \
    --model_path ./models/RSMP \
    --output_path ./outputs/RSMP \
    --log_path ./logs/RSMP \
    --gpu_id ${GPU_ID}
```

对`eval.sh`进行相同的修改，确保也包含`--gpu_id ${GPU_ID}`参数。

## 4. 验证修改

创建一个简单的GPU检查脚本`check_gpu.py`：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查GPU_ID参数是否正确传递
"""

import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='检查GPU_ID参数')
    parser.add_argument('--gpu_id', type=int, default=1, help='使用的GPU设备序号 (0, 1, 2, ...)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"命令行参数中的GPU_ID: {args.gpu_id}")
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    
    # 检查CUDA是否可用
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"当前CUDA设备名称: {torch.cuda.get_device_name(args.gpu_id)}")
```

验证步骤：

```bash
# 检查GPU设置是否生效
python check_gpu.py --gpu_id 2

# 运行训练脚本
bash train.sh

# 运行评估脚本
bash eval.sh
```

## 5. 总结

通过以上一步到位的修改，我们解决了所有与GPU_ID相关的问题。现在可以通过简单修改`train.sh`和`eval.sh`中的`GPU_ID`参数来切换GPU设备，无需进行其他修改。

此修改方案的优势：
1. **一步到位**：无需多个修复脚本，直接解决所有问题
2. **灵活性**：可以轻松切换GPU设备而无需修改代码
3. **可维护性**：统一的设备管理，避免硬编码
4. **资源优化**：根据实际需求动态分配GPU资源