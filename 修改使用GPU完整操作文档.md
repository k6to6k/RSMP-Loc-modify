# GPU_ID 修改完整操作文档

## 1. 背景介绍

在深度学习项目中，特别是有多 GPU 环境的服务器上，正确设置和切换 GPU 设备是非常重要的。本文档记录了在 RSMP 项目中修改 GPU_ID 参数时遇到的问题及解决方案。

## 2. 问题概述

在项目中发现了以下与 GPU 设备相关的问题：

1. **训练脚本问题**：在 `train.sh` 中设置 `GPU_ID=2`，但实际训练时仍然使用 1 号 GPU，导致无法按预期切换 GPU。

2. **评估脚本问题**：在 `eval.sh` 中运行评估脚本时，出现模型加载错误：
   ```
   RuntimeError: Attempting to deserialize object on CUDA device 2 but torch.cuda.device_count() is 2.
   ```

## 3. 问题分析

通过代码审查，发现了以下具体问题：

### 3.1 训练脚本问题

1. **device 变量引用错误**：
   - 在 `train.py` 文件中使用了 `device` 变量，但该变量并未在函数中定义
   - 正确的引用应该是 `config.device`，这个变量在 `config.py` 中定义

2. **config 变量未定义**：
   - 在 `Total_loss` 和 `Total_loss_Gai` 类的 `forward` 方法中使用了 `config` 变量
   - 但该变量并未作为参数传入这些方法

### 3.2 评估脚本问题

1. **模型加载缺少 map_location 参数**：
   - 在 `main_eval.py` 中加载模型时没有指定 `map_location` 参数
   - 导致无法将模型从一个 GPU 设备映射到另一个设备
   - 当模型在 GPU 2 上保存，但尝试在 GPU 1 上加载时会出错

## 4. 解决方案

### 4.1 创建修复脚本

我们创建了一个主要脚本和三个辅助修复脚本来解决上述问题：

1. **replace_gpu.py**：主要脚本，替换所有文件中的硬编码GPU设备引用
2. **fix_train.py**：辅助脚本，修复 `train.py` 中的 `device` 引用问题
3. **fix_config_param.py**：辅助脚本，修复 `train.py` 中的 `config` 变量未定义问题
4. **fix_eval.py**：辅助脚本，修复 `main_eval.py` 中的模型加载问题

### 4.2 修复训练脚本

#### 4.2.1 修复 device 引用

在 `fix_train.py` 中，我们将 `train.py` 中所有的 `.to(device)` 改为 `.to(config.device)`：

```python
replacements = [
    ('.to(device)', '.to(config.device)'),
    ('torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(device)', 
     'torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(config.device)'),
    # 更多替换...
]

for old_str, new_str in replacements:
    replace_in_file(train_file, old_str, new_str)
```

#### 4.2.2 添加 config 参数

在 `fix_config_param.py` 中，我们修改了 `Total_loss` 和 `Total_loss_Gai` 类的 `forward` 方法，添加一个可选的 `config` 参数：

```python
# 修改Total_loss类的forward方法
pattern1 = r'def forward\(self, vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, label, point_anno, step, pseudo_instance_label, uncertainty\):'
replacement1 = 'def forward(self, vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, label, point_anno, step, pseudo_instance_label, uncertainty, config=None):'
```

同时，修改 `train` 函数中的 `criterion` 调用，传递 `config` 参数：

```python
pattern3 = r'cost, loss = criterion\(vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, _label, _point_anno, step, pseudo_instance_label=pseudo_instance_label, uncertainty=uncertainty\)'
replacement3 = 'cost, loss = criterion(vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, _label, _point_anno, step, pseudo_instance_label=pseudo_instance_label, uncertainty=uncertainty, config=config)'
```

### 4.3 修复评估脚本

在 `fix_eval.py` 中，我们修改了 `main_eval.py` 中的模型加载代码，添加 `map_location` 参数：

```python
pattern = r'net\.load_state_dict\(torch\.load\(args\.model_file\)\)'
replacement = 'net.load_state_dict(torch.load(args.model_file, map_location=config.device))'
```

## 5. 应用修复步骤

按照以下步骤应用所有修复：

1. 首先运行 `python replace_gpu.py` 替换所有文件中的硬编码GPU设备引用：
   ```bash
   python replace_gpu.py
   ```

2. 如果 replace_gpu.py 无法解决所有问题，则运行以下辅助修复脚本：

   a. 运行 `python fix_train.py` 修复 device 引用问题：
   ```bash
   python fix_train.py
   ```

   b. 运行 `python fix_config_param.py` 修复 config 变量未定义问题：
   ```bash
   python fix_config_param.py
   ```

   c. 运行 `python fix_eval.py` 修复评估脚本中的模型加载问题：
   ```bash
   python fix_eval.py
   ```

## 6. 验证修复

### 6.1 验证 GPU 设置

创建并运行 `check_gpu.py` 脚本来验证 GPU 设置是否生效：

```bash
python check_gpu.py --gpu_id 2
```

正确的输出应显示：
```
命令行参数中的GPU_ID: 2
使用的设备: cuda:2
CUDA是否可用: True
CUDA设备数量: 8
当前CUDA设备: 0
当前CUDA设备名称: NVIDIA GeForce RTX 4090
```

### 6.2 修改训练和评估脚本

需要修改 `train.sh` 和 `eval.sh` 文件中的两处内容：

1. **设置 GPU_ID 变量**：
   ```bash
   # 修改为所需的GPU ID（例如：2）
   GPU_ID=2
   ```

2. **设置 CUDA_VISIBLE_DEVICES**：
   将原来的限制性设置修改为服务器上所有可用的GPU：
   ```bash
   # 修改前
   # CUDA_VISIBLE_DEVICES=0,1,2
   
   # 修改后（假设服务器有8个GPU）
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ```

3. **确保命令行参数包含 gpu_id**：
   ```bash
   python main.py --gpu_id ${GPU_ID} [其他参数...]
   ```

示例修改后的 `train.sh` 文件：

```bash
#!/bin/bash

# 设置要使用的GPU ID
GPU_ID=2

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

### 6.3 验证修复

运行修改后的脚本：

```bash
bash train.sh
bash eval.sh
```

训练和评估应该在指定的 GPU 上进行，不再出现之前的错误。

## 7. 修复文件说明

### 7.1 replace_gpu.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
替换所有文件中的硬编码GPU设备引用，将torch.device('cuda:1')改为config.device
"""

import os
import re

def find_python_files(directory):
    """查找目录中的所有Python文件"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def replace_in_file(file_path):
    """在文件中替换硬编码的GPU设备引用"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 替换直接使用的torch.device('cuda:1')
        pattern1 = r"torch\.device\(['\"]cuda:1['\"]\)"
        replacement1 = "config.device"
        
        # 替换直接使用的'cuda:1'字符串
        pattern2 = r"['\"]cuda:1['\"]\s*[,)]" 
        replacement2 = "config.device\1"
        
        # 替换.to('cuda:1')和.to("cuda:1")
        pattern3 = r"\.to\(['\"]cuda:1['\"]\)"
        replacement3 = ".to(config.device)"
        
        # 应用替换
        new_content = re.sub(pattern1, replacement1, content)
        new_content = re.sub(pattern2, lambda m: m.group(0).replace("'cuda:1'", "config.device").replace('"cuda:1"', "config.device"), new_content)
        new_content = re.sub(pattern3, ".to(config.device)", new_content)
        
        # 检查是否有修改
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            return True, file_path
        return False, file_path
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False, file_path

if __name__ == "__main__":
    # 当前目录
    current_dir = os.getcwd()
    
    # 查找所有Python文件
    python_files = find_python_files(current_dir)
    print(f"找到 {len(python_files)} 个Python文件")
    
    # 替换文件中的硬编码GPU设备引用
    modified_files = []
    for file_path in python_files:
        modified, path = replace_in_file(file_path)
        if modified:
            modified_files.append(path)
    
    print(f"已修改 {len(modified_files)} 个文件:")
    for file_path in modified_files:
        print(f"  - {file_path}")
```

### 7.2 fix_train.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复train.py中的device引用问题
"""

import os

def replace_in_file(file_path, old_str, new_str):
    """替换文件中的字符串"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # 检查文件中是否包含旧字符串
        if old_str in content:
            content = content.replace(old_str, new_str)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"已更新: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False

# 修复train.py中的device引用
train_file = 'train.py'
replacements = [
    ('.to(device)', '.to(config.device)'),
    ('torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(device)', 'torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(config.device)'),
    ('torch.zeros((new_dense_anno.shape[0], new_dense_anno.shape[1], 1)).to(device)', 'torch.zeros((new_dense_anno.shape[0], new_dense_anno.shape[1], 1)).to(config.device)'),
    ('torch.zeros_like(point_anno).to(device)', 'torch.zeros_like(point_anno).to(config.device)'),
    ('torch.tensor(label_lst).to(device)', 'torch.tensor(label_lst).to(config.device)'),
    ('sim_matrix = torch.exp(sim_matrix).to(device)', 'sim_matrix = torch.exp(sim_matrix).to(config.device)'),
    ('sim_matrix = sim_matrix.clone().fill_diagonal_(0).to(device)', 'sim_matrix = sim_matrix.clone().fill_diagonal_(0).to(config.device)'),
    ('sim_matrix = sim_matrix.clone().fill_diagonal_(0)', 'sim_matrix = sim_matrix.clone().fill_diagonal_(0).to(config.device)')
]

print("正在修复train.py中的device引用问题...")
fixed = False
for old_str, new_str in replacements:
    if replace_in_file(train_file, old_str, new_str):
        fixed = True

if fixed:
    print("修复完成！现在train.py将正确使用config.device参数。")
else:
    print("未找到需要修复的内容，可能已经修复或文件结构发生变化。")
```

### 7.3 fix_config_param.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复train.py中的config变量未定义问题
"""

import os
import re

def modify_file(file_path):
    """修改文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 修改Total_loss类的forward方法
        pattern1 = r'def forward\(self, vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, label, point_anno, step, pseudo_instance_label, uncertainty\):'
        replacement1 = 'def forward(self, vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, label, point_anno, step, pseudo_instance_label, uncertainty, config=None):'
        
        # 修改Total_loss_Gai类的forward方法
        pattern2 = r'def forward\(self, vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, label, point_anno, step, pseudo_instance_label, uncertainty\):'
        replacement2 = 'def forward(self, vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, label, point_anno, step, pseudo_instance_label, uncertainty, config=None):'
        
        # 修改train函数中的criterion调用
        pattern3 = r'cost, loss = criterion\(vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, _label, _point_anno, step, pseudo_instance_label=pseudo_instance_label, uncertainty=uncertainty\)'
        replacement3 = 'cost, loss = criterion(vid_score, cas_sigmoid_fuse, features, cls_att, stored_info, _label, _point_anno, step, pseudo_instance_label=pseudo_instance_label, uncertainty=uncertainty, config=config)'
        
        # 应用替换
        content = re.sub(pattern1, replacement1, content, count=1)
        content = re.sub(pattern2, replacement2, content, count=1)
        content = re.sub(pattern3, replacement3, content, count=1)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"已更新: {file_path}")
        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False

print("正在修复train.py中的config变量未定义问题...")
if modify_file('train.py'):
    print("修复完成！现在train.py可以正确接收config参数。")
else:
    print("修复失败，请检查文件路径或内容。")
```

### 7.4 fix_eval.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复main_eval.py中的模型加载问题
"""

import os
import re

def modify_file(file_path):
    """修改文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 修改模型加载代码，添加map_location参数
        pattern = r'net\.load_state_dict\(torch\.load\(args\.model_file\)\)'
        replacement = 'net.load_state_dict(torch.load(args.model_file, map_location=config.device))'
        
        # 应用替换
        new_content = re.sub(pattern, replacement, content)
        
        # 检查是否有修改
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"已更新: {file_path}")
            return True
        else:
            print(f"未找到需要修改的内容: {file_path}")
            return False
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False

print("正在修复main_eval.py中的模型加载问题...")
if modify_file('main_eval.py'):
    print("修复完成！现在main_eval.py可以正确加载模型到指定的GPU设备。")
else:
    print("修复失败，请检查文件路径或内容。")
```

### 7.5 check_gpu.py

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

## 8. 完整修改流程示例

以下是一个完整的修改流程示例，假设需要将GPU从1号改为2号：

1. **修改配置文件**：
   - 确保 `config.py` 中正确设置了 `device` 属性
   - 确保 `options.py` 中添加了 `--gpu_id` 参数

2. **修改训练脚本**：
   - 编辑 `train.sh`，将 `GPU_ID=1` 改为 `GPU_ID=2`
   - 设置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
   - 确保命令行参数中包含 `--gpu_id ${GPU_ID}`

3. **修改评估脚本**：
   - 编辑 `eval.sh`，将 `GPU_ID=1` 改为 `GPU_ID=2`
   - 设置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
   - 确保命令行参数中包含 `--gpu_id ${GPU_ID}`

4. **运行修复脚本**：
   ```bash
   # 首先替换所有文件中的硬编码GPU设备引用
   python replace_gpu.py
   
   # 如果需要，运行以下辅助修复脚本
   python fix_train.py
   python fix_config_param.py
   python fix_eval.py
   ```

5. **验证修复**：
   ```bash
   # 检查GPU设置是否生效
   python check_gpu.py --gpu_id 2
   
   # 运行训练脚本
   bash train.sh
   
   # 运行评估脚本
   bash eval.sh
   ```

## 9. 总结

通过以上修复，我们解决了项目中与 GPU_ID 参数相关的所有问题。现在可以通过修改 `train.sh` 和 `eval.sh` 中的 `GPU_ID` 参数来正确选择使用的 GPU 设备，无论是训练还是评估都能在指定的 GPU 上运行。

这些修复主要涉及四个方面：
1. 修复变量引用错误
2. 添加缺失的函数参数
3. 添加模型加载时的设备映射
4. 替换硬编码的GPU设备引用

这种动态GPU配置方法的优势：
1. **灵活性**：无需修改代码即可切换GPU设备
2. **并行实验**：可以同时在不同GPU上运行不同参数的实验
3. **资源利用**：根据实际负载动态分配GPU资源
4. **可维护性**：统一的设备管理，避免硬编码

这些经验对于其他深度学习项目中的 GPU 设备管理也有参考价值。
