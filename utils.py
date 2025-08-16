import re
import torch
import numpy as np
from typing import Union
def parse_range_or_single(input_str):
    """
    解析输入字符串，支持以下格式：
    - 单个值: "5" -> [5]
    - 区间: "[1,10]" -> [1,2,3,4,5,6,7,8,9,10]
    - 带步长区间: "[1,10:2]" -> [1,3,5,7,9]
    - 列表: "{1,3,5}" -> [1,3,5]
    """
    input_str = input_str.strip()
    
    # 如果是区间格式 [start,end]
    range_match = re.match(r'^\[(\d+),(\d+)\]$', input_str)
    if range_match:
        start, end = map(int, range_match.groups())
        return list(range(start, end + 1))
    
    # 如果是带步长的区间格式 [start,end:step]
    range_step_match = re.match(r'^\[(\d+),(\d+):(\d+)\]$', input_str)
    if range_step_match:
        start, end, step = map(int, range_step_match.groups())
        return list(range(start, end + 1, step))
    
    # 如果是列表格式 {1,3,5,7}
    list_match = re.match(r'^\{(.+)\}$', input_str)
    if list_match:
        values_str = list_match.group(1)
        return [int(x.strip()) for x in values_str.split(',')]
    
    # 如果是单个数字
    if input_str.isdigit():
        return [int(input_str)]
    
    # 如果都不匹配，抛出错误
    raise ValueError(f"无法解析输入格式: {input_str}. 支持的格式: '5'(单个), '[1,10]'(区间), '{{1,3,5}}'(列表)")

def list_to_range(input_list):
    """
    将列表转换为范围格式 [start, end]
    
    参数:
        input_list: 输入列表，如 [1,3,5,7,9] 或 [5] 或 [1,2,3,4,5]
        
    返回:
        [start, end + 1): 范围格式，包含start和end
        
    示例:
        [1,3,5,7,9] -> [1, 10)
        [5] -> [5, 6)  
        [1,2,3,4,5] -> [1, 6)
        [] -> 抛出异常
    """
    if not input_list:
        raise ValueError("输入列表不能为空")
    
    if not isinstance(input_list, list):
        raise TypeError("输入必须是列表类型")
    
    # 确保列表中的元素都是数字
    try:
        numeric_list = [int(x) for x in input_list]
    except (ValueError, TypeError):
        raise ValueError("列表中的所有元素必须是数字")
    
    # 对列表进行排序，确保顺序正确
    sorted_list = sorted(numeric_list)
    
    if len(sorted_list) == 1:
        return [sorted_list[0], sorted_list[0] + 1]
    else:
        return [sorted_list[0], sorted_list[-1] + 1]

def validate_range(range_list, name="range"):
    """
    验证范围是否有效
    
    参数:
        range_list: [start, end] 格式的范围
        name: 范围的名称，用于错误信息
    """
    if not isinstance(range_list, list) or len(range_list) != 2:
        raise ValueError(f"{name} 必须是包含两个元素的列表: [start, end]")
    
    start, end = range_list
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{name} 的元素必须是整数")
    
    if start < 0 or end < 0:
        raise ValueError(f"{name} 的值必须非负: [{start}, {end}]")
    
    if start > end:
        raise ValueError(f"{name} 的起始值不能大于结束值: [{start}, {end}]")
    
    return True

def print_range_info(cycle_range, scene_range, dataset_type=""):
    """
    打印范围信息，用于调试
    """
    print(f"{dataset_type} cycle range: {cycle_range}")
    print(f"{dataset_type} scene range: {scene_range}")
    expected_samples = (cycle_range[1] - cycle_range[0] + 1) * (scene_range[1] - scene_range[0] + 1)
    print(f"{dataset_type} expected samples: {expected_samples}")
