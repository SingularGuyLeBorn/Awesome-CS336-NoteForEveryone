import re
import os
def split_markdown_file(input_file_path: str, output_directory: str):
    """
    读取一个包含多个Markdown文件内容的文本文件,并将其分割成独立的.md文件. 
    文件内容块由以下格式的分隔符定义: 
    --- FILE: [filename.md] ---
    (文件内容...)
    --- END OF FILE ---
    Args:
        input_file_path (str): 包含所有内容的源文本文件的路径. 
        output_directory (str): 用于存放生成的.md文件的目录. 
    """
    print(f"开始处理文件: {input_file_path}")
    # 1. 检查并创建输出目录
    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"输出目录 '{output_directory}' 已准备好. ")
    except OSError as e:
        print(f"错误: 无法创建目录 {output_directory}. 原因: {e}")
        return
    # 2. 读取源文件内容
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_file_path}")
        return
    except Exception as e:
        print(f"错误: 读取文件时出错. 原因: {e}")
        return
    # 3. 使用正则表达式查找所有文件块
    # 正则表达式解释:
    # --- FILE: (.*?) ---\n  : 匹配开始分隔符,并捕获文件名 (非贪婪模式)
    # (.*?)                  : 捕获两个分隔符之间的所有内容,包括换行符 (re.DOTALL)
    # \n--- END OF FILE ---    : 匹配结束分隔符
    pattern = r"--- FILE: (.*?) ---\n(.*?)\n--- END OF FILE ---"
    
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        print("警告: 在输入文件中没有找到任何有效的文件块. 请检查文件格式是否正确. ")
        return
    print(f"找到了 {len(matches)} 个文件块. 正在生成文件...")
    # 4. 遍历所有匹配项并创建文件
    for filename, file_content in matches:
        # 清理文件名和内容中可能存在的多余空白
        clean_filename = filename.strip()
        clean_content = file_content.strip()
        
        output_path = os.path.join(output_directory, clean_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.write(clean_content)
            print(f"  [成功] -> 已创建文件: {output_path}")
        except IOError as e:
            print(f"  [失败] -> 无法写入文件 {output_path}. 原因: {e}")
            
    print("\n处理完成！")
if __name__ == '__main__':
    # --- 用户配置 ---
    # 1. 将我生成的全部内容保存到一个文本文件中,例如 "generated_lecture_notes.txt"
    input_filename = "./Lecture7-Main.md"
    # 2. 指定一个目录名,用于存放所有生成的 .md 文件
    output_dir = "./"
    # --- 运行脚本 ---
    split_markdown_file(input_filename, output_dir)