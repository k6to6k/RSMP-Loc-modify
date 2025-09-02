import re

def fix_frm_scr_issue():
    """修复frm_scr在动态原型模式下未定义的问题"""
    
    # 读取model.py文件
    with open('model.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找目标代码段
    pattern = r'(if use_dynamic_prototype and context_vector is not None:.*?for b in range\(B\):.*?ca_vid_scr\[b\].*?cw_vid_scr\[b\].*?)(\s+else:)'
    
    # 检查匹配结果
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # 构建替换字符串
        replacement = match.group(1) + '\n            # 动态原型模式下，明确指定frm_scr\n            frm_scr = frm_scrs' + match.group(2)
        
        # 执行替换
        modified_content = content.replace(match.group(0), replacement)
        
        # 写回文件
        with open('model.py', 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print("成功修复model.py中的frm_scr问题")
        return True
    else:
        print("未找到匹配的代码段，请手动修改")
        return False

# 运行修复函数
if __name__ == "__main__":
    fix_frm_scr_issue()

