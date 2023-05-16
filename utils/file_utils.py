# 如何使用python遍历一个目录，得到树结构？
# 我想要更加规范的DST结构，而不是字符串，比如{'a.txt':None,'dirA':{'b.txt','c.py'}}

import os
import time


def build_tree(dir_path):
    tree = {}
    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isdir(path):
            tree[filename] = build_tree(path)
        else:
            tree[filename] = None
    return tree


def get_file_suffix(file_name):
    return os.path.splitext(file_name)[-1]


def convert_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return "%3.1f %s" % (size_bytes, unit)
        size_bytes /= 1024.0
    return "%3.1f %s" % (size_bytes, 'PB')


def get_file_info(path):
    info = os.stat(path)
    size = convert_size(info.st_size)
    print(info)
    mtime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(info.st_mtime))
    return (size, mtime)
