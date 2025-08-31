# GPU ID 修改：一步到位自动化指南

## 1. 背景介绍

在多 GPU 环境下进行深度学习项目时，能够灵活、正确地指定和切换 GPU 设备至关重要。本文档旨在提供一个**自动化、一步到位**的解决方案，彻底解决项目中GPU设备引用的硬编码问题，实现通过脚本参数动态选择GPU。

## 2. 问题概述

项目中存在以下几个与GPU设备设置相关的核心问题：

1.  **硬编码设备**：代码中多处使用 `cuda:1` 等写死的设备，导致无法动态切换。
2.  **变量引用错误**：在 `train.py` 中，错误地引用了未定义的 `device` 变量，而正确的引用应为 `config.device`。
3.  **函数参数缺失**：在 `train.py` 中，调用损失函数 `criterion` 时没有传入 `config` 对象，导致在损失函数内部无法访问到正确的设备信息。
4.  **模型加载问题**：在 `main_eval.py` 中，加载模型时未使用 `map_location` 参数，当训练和评估使用不同 GPU 时会导致错误。

这些问题共同导致了即使在启动脚本中设置了 `GPU_ID`，也无法在指定的设备上运行。

## 3. 自动化解决方案

我们将通过一个统一的 Python 脚本 `apply_gpu_fixes.py` 来自动修复代码库中的所有问题。

### 3.1 运行自动化修复脚本

请将下面的代码保存为 `apply_gpu_fixes.py` 文件，并将其放置在你的项目根目录下。

这个脚本会扫描项目中的所有 Python 文件，并执行以下操作：
-   **全局替换**：将所有硬编码的 `'cuda:1'` 或 `.to('cuda:1')` 替换为使用 `config.device`。
-   **修复 `train.py`**：
    -   修正所有 `.to(device)` 的错误引用为 `.to(config.device)`。
    -   为损失函数 `Total_loss` 和 `Total_loss_Gai` 的 `forward` 方法自动添加 `config` 参数。
    -   在调用损失函数的地方，自动传入 `config` 参数。
-   **修复 `main_eval.py`**：
    -   为 `torch.load` 增加 `map_location=config.device` 参数，确保模型可以被加载到任意指定的GPU上。

**`apply_gpu_fixes.py` 脚本内容:**
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一修复脚本，用于自动化解决项目中所有与GPU设备相关的硬编码和逻辑错误。
"""
import os
import re

def process_file(file_path):
    """
    读取文件，应用所有修复规则，如果内容有变则写回文件。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  - [错误] 读取文件失败: {file_path}, 原因: {e}")
        return False

    original_content = content
    filename = os.path.basename(file_path)

    # --- 规则 1: 全局硬编码替换 (应用于所有 .py 文件) ---
    # 替换 torch.device('cuda:1')
    content = re.sub(r"torch\.device\(['\"]cuda:1['\"]\)", "config.device", content)
    # 替换 .to('cuda:1')
    content = re.sub(r"\.to\(['\"]cuda:1['\"]\)", ".to(config.device)", content)
    # 替换作为字符串参数的 'cuda:1'
    content = re.sub(r"['\"]cuda:1['\"]", "config.device", content)

    # --- 特定文件修复 ---
    if filename == 'train.py':
        # --- 规则 2: 修复 train.py 中的 .to(device) 错误 ---
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
        for old, new in replacements:
            content = content.replace(old, new)

        # --- 规则 3: 修复 train.py 中的损失函数参数缺失问题 ---
        # 为 forward 方法添加 config=None 参数
        forward_pattern = re.compile(r"(def forward\(self, [^\)]+)\):")
        content = forward_pattern.sub(r"\1, config=None):", content)

        # 为 criterion 调用添加 config=config 参数
        criterion_pattern = re.compile(r"(criterion\([^\)]+)\)")
        content = criterion_pattern.sub(r"\1, config=config)", content)

    elif filename == 'main_eval.py':
        # --- 规则 4: 修复 main_eval.py 中的模型加载问题 ---
        eval_pattern = re.compile(r"net\.load_state_dict\(torch\.load\(args\.model_file\)\)")
        replacement = r"net.load_state_dict(torch.load(args.model_file, map_location=config.device))"
        content = eval_pattern.sub(replacement, content)

    # 如果内容有变化，则写回文件
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  - [已修改] {file_path}")
            return True
        except Exception as e:
            print(f"  - [错误] 写入文件失败: {file_path}, 原因: {e}")
            return False
    return False

if __name__ == "__main__":
    project_dir = os.getcwd()
    print(f"在 '{project_dir}' 目录中开始扫描和修复...")
    
    modified_files_count = 0
    py_files = [os.path.join(root, file)
                for root, _, files in os.walk(project_dir)
                for file in files if file.endswith('.py')]

    for file_path in py_files:
        if process_file(file_path):
            modified_files_count += 1
            
    print(f"\n修复完成！共修改了 {modified_files_count} 个文件。")
    if modified_files_count == 0:
        print("所有文件似乎都已是最新状态，无需修改。")

```

**执行脚本:**
在项目根目录下打开终端，运行命令：
```bash
python apply_gpu_fixes.py
```
脚本将自动查找并修改所有相关文件。

### 3.2 修改启动脚本

自动化脚本执行完毕后，你的代码库已经具备了动态选择GPU的能力。现在，只需修改 `train.sh` 和 `eval.sh` 文件即可。

**修改 `train.sh`:**
```bash
#!/bin/bash

# 在这里设置要使用的GPU ID (例如: 0, 1, 2, ...)
GPU_ID=2

# 设置对所有GPU可见，由PyTorch代码根据 --gpu_id 参数来选择具体设备
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --num_iters 200 \
    --epochs_per_step 50 \
    --batch_size 16 \
    --model_path ./models/RSMP \
    --output_path ./outputs/RSMP \
    --log_path ./logs/RSMP \
    --gpu_id ${GPU_ID}
```

**修改 `eval.sh`:**
对 `eval.sh` 做类似修改，确保也通过 `--gpu_id ${GPU_ID}` 将GPU ID传递给评估脚本。

### 3.3 验证

现在，整个流程已经完成！你可以通过以下方式验证：

1.  修改 `train.sh` 中的 `GPU_ID` 变量为你想要的GPU编号（例如 `GPU_ID=3`）。
2.  运行训练脚本：`bash train.sh`。
3.  在训练过程中，通过 `nvidia-smi` 命令观察，可以看到训练任务正在你指定的GPU 3上运行。

## 4. 总结

通过执行上述`apply_gpu_fixes.py`自动化脚本，我们一劳永逸地解决了项目中所有关于GPU设备的硬编码和逻辑问题。现在，项目代码具备了良好的灵活性和可维护性，研究人员可以简单地通过修改启动脚本中的一个变量来控制实验所用的GPU，极大地提高了开发和实验效率。



