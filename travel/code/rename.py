import os
import re

# 需要修改文件名的文件夹路径
dir_path = 'source/travel/photo/'

# 遍历文件夹
for filename in os.listdir(dir_path):
    # 使用正则表达式匹配要替换的部分
    new_filename = re.sub(r'-\d+\.jpg$', '.jpg', filename)
    
    # 如果文件名需要修改
    if new_filename != filename:
        # 获取完整的原文件路径和新文件路径
        old_file_path = os.path.join(dir_path, filename)
        new_file_path = os.path.join(dir_path, new_filename)
        
        # 重命名文件
        os.rename(old_file_path, new_file_path)
