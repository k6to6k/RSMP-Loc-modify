import torch
import numpy as np
from model import Model_GAI, ContextPrototypeModulator

def test_context_prototype_modulator():
    """测试ContextPrototypeModulator模块"""
    print("===== 测试上下文原型调制网络 =====")
    feature_dim = 2048
    batch_size = 2
    num_classes = 20
    
    # 创建模块实例
    modulator = ContextPrototypeModulator(feature_dim=feature_dim)
    
    # 创建测试数据
    prototype = torch.randn(num_classes + 1, feature_dim)  # 全局原型
    context_vector = torch.randn(batch_size, feature_dim)  # 上下文向量
    
    # 运行模块
    dynamic_prototype = modulator(prototype, context_vector)
    
    # 验证输出形状
    print(f"输入原型形状: {prototype.shape}")
    print(f"上下文向量形状: {context_vector.shape}")
    print(f"动态原型形状: {dynamic_prototype.shape}")
    assert dynamic_prototype.shape == (batch_size, num_classes + 1, feature_dim)
    
    # 验证输出是否包含合理的偏移
    delta = dynamic_prototype[0] - prototype
    print(f"原型调整量平均值: {delta.mean().item():.6f}")
    print(f"原型调整量标准差: {delta.std().item():.6f}")
    
    print("上下文原型调制网络测试通过!\n")
    return True

def test_prototype_memory_update():
    """测试原型记忆库更新机制"""
    print("===== 测试原型记忆库更新 =====")
    len_feature = 2048
    num_classes = 20
    batch_size = 2
    
    # 创建模型实例
    model = Model_GAI(len_feature, num_classes, r_act=8)
    
    # 确保处于训练模式
    model.train()
    
    # 创建测试数据
    features = torch.randn(batch_size, len_feature)  # 特征
    labels = torch.zeros(batch_size, num_classes)  # 标签
    labels[0, 5] = 1  # 第一个样本是类别5
    labels[1, 10] = 1  # 第二个样本是类别10
    
    # 记录原始原型
    old_prototype = model.prototype_memory.clone()
    
    # 更新原型
    model.update_prototype_memory(features, labels)
    
    # 验证原型是否更新
    updated_classes = [5, 10]
    for c in updated_classes:
        is_changed = not torch.allclose(old_prototype[c], model.prototype_memory[c])
        print(f"类别 {c} 原型已更新: {is_changed}")
        assert is_changed
    
    # 验证其他类别原型保持不变
    for c in range(num_classes):
        if c not in updated_classes:
            is_unchanged = torch.allclose(old_prototype[c], model.prototype_memory[c])
            assert is_unchanged
    
    print("原型记忆库更新测试通过!\n")
    return True

def test_extract_context_vector():
    """测试上下文向量提取"""
    print("===== 测试上下文向量提取 =====")
    len_feature = 2048
    num_classes = 20
    batch_size = 2
    seq_len = 10
    
    # 创建模型实例
    model = Model_GAI(len_feature, num_classes, r_act=8)
    
    # 创建测试数据
    features = torch.randn(batch_size, seq_len, len_feature)  # [B, T, D]
    
    # 提取上下文向量
    context_vector = model.extract_context_vector(features)
    
    # 验证输出形状
    print(f"特征形状: {features.shape}")
    print(f"上下文向量形状: {context_vector.shape}")
    assert context_vector.shape == (batch_size, len_feature)
    
    # 验证上下文向量是否是时间维度的平均
    manual_context = torch.mean(features, dim=1)
    is_equal = torch.allclose(context_vector, manual_context)
    print(f"上下文向量是特征时间维度平均: {is_equal}")
    assert is_equal
    
    print("上下文向量提取测试通过!\n")
    return True

def test_prediction_module():
    """测试预测模块与动态原型的集成"""
    print("===== 测试预测模块 =====")
    len_feature = 2048
    num_classes = 20
    batch_size = 2
    seq_len = 10
    
    # 创建模型实例
    model = Model_GAI(len_feature, num_classes, r_act=8)
    
    # 创建测试数据
    features = torch.randn(batch_size, seq_len, len_feature)  # [B, T, D]
    context_vector = model.extract_context_vector(features)
    
    # 测试静态原型
    print("测试静态原型模式...")
    try:
        static_pred = model.PredictionModule(features, context_vector=None, use_dynamic_prototype=False)
        print(f"静态原型预测结果形状:")
        print(f"- vid_ca_pred: {static_pred[0].shape}")
        print(f"- vid_cw_pred: {static_pred[1].shape}")
        print(f"- class_agno_att: {static_pred[2].shape}")
        print(f"- frm_scr: {static_pred[3].shape}")
        static_ok = True
    except Exception as e:
        print(f"静态原型测试失败: {str(e)}")
        static_ok = False
    
    # 测试动态原型
    print("测试动态原型模式...")
    try:
        dynamic_pred = model.PredictionModule(features, context_vector, use_dynamic_prototype=True)
        print(f"动态原型预测结果形状:")
        print(f"- vid_ca_pred: {dynamic_pred[0].shape}")
        print(f"- vid_cw_pred: {dynamic_pred[1].shape}")
        print(f"- class_agno_att: {dynamic_pred[2].shape}")
        print(f"- frm_scr: {dynamic_pred[3].shape}")
        dynamic_ok = True
    except Exception as e:
        print(f"动态原型测试失败: {str(e)}")
        dynamic_ok = False
    
    # 验证输出形状是否相同
    if static_ok and dynamic_ok:
        assert static_pred[0].shape == dynamic_pred[0].shape
        assert static_pred[1].shape == dynamic_pred[1].shape
        
        # 验证两种模式的输出是否不同
        is_different = not torch.allclose(static_pred[0], dynamic_pred[0])
        print(f"静态与动态原型预测结果不同: {is_different}")
    
    print("预测模块测试通过!\n")
    return static_ok and dynamic_ok

def test_forward_pass():
    """测试前向传播"""
    print("===== 测试前向传播 =====")
    len_feature = 2048
    num_classes = 20
    batch_size = 1
    seq_len = 16
    
    # 创建模型实例
    model = Model_GAI(len_feature, num_classes, r_act=8)
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, len_feature)  # 输入特征
    vid_labels = torch.zeros(batch_size, num_classes)  # 视频标签
    vid_labels[0, 5] = 1  # 类别5
    
    try:
        # 前向传播
        outputs = model(x, vid_labels)
        
        # 验证输出
        vid_scores, cas_sigmoid_fuse, features, pred_outputs, _ = outputs
        
        print(f"视频分数形状: {vid_scores[0].shape}")
        print(f"CAS输出形状: {cas_sigmoid_fuse[0].shape}")
        print(f"特征形状: {features.shape}")
        print(f"预测输出数量: {len(pred_outputs)}")
        
        # 检查o_vid_ca_pred和m_vid_ca_pred
        o_vid_ca_pred, m_vid_ca_pred = pred_outputs[0], pred_outputs[1]
        print(f"o_vid_ca_pred形状: {o_vid_ca_pred.shape}")
        print(f"m_vid_ca_pred形状: {m_vid_ca_pred.shape}")
        
        print("前向传播测试通过!\n")
        return True
    except Exception as e:
        print(f"前向传播测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 运行所有测试
        tests = [
            test_context_prototype_modulator,
            test_prototype_memory_update,
            test_extract_context_vector,
            test_prediction_module,
            test_forward_pass
        ]
        
        all_passed = True
        for test_func in tests:
            try:
                result = test_func()
                all_passed = all_passed and result
            except Exception as e:
                print(f"测试失败: {test_func.__name__}")
                print(f"错误信息: {str(e)}")
                all_passed = False
        
        if all_passed:
            print("所有测试通过！上下文感知的动态原型模块功能正常。")
        else:
            print("有测试失败，请检查错误信息。")
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
