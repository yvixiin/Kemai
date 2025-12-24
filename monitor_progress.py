#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时监控批量处理进度"""

import json
import time
import os
from pathlib import Path
from collections import defaultdict

def get_progress_stats():
    """获取进度统计"""
    output_dir = Path('e:/kemai/category_output')

    # 读取进度文件
    progress_file = output_dir / 'progress.json'
    if not progress_file.exists():
        return None

    with open(progress_file, 'r', encoding='utf-8') as f:
        progress = json.load(f)

    completed = progress.get('completed', [])
    failed = progress.get('failed', {})

    # 统计总品类数
    input_dir = Path('e:/kemai/category_csv')
    total_categories = len(list(input_dir.glob('*.csv')))

    # 统计各组别
    groups_stats = defaultdict(int)
    for group_dir in output_dir.iterdir():
        if group_dir.is_dir() and group_dir.name != 'cache':
            csv_files = list(group_dir.glob('*_result.csv'))
            groups_stats[group_dir.name] = len(csv_files)

    # 统计总条目数
    total_items = 0
    for group_dir in output_dir.iterdir():
        if group_dir.is_dir() and group_dir.name != 'cache':
            for csv_file in group_dir.glob('*_result.csv'):
                try:
                    with open(csv_file, 'r', encoding='utf-8-sig', errors='ignore') as f:
                        total_items += sum(1 for _ in f) - 1  # 减去表头
                except:
                    pass

    return {
        'total_categories': total_categories,
        'completed_count': len(completed),
        'failed_count': len(failed),
        'remaining': total_categories - len(completed) - len(failed),
        'groups_stats': groups_stats,
        'total_items': total_items,
        'recently_completed': completed[-5:] if len(completed) >= 5 else completed
    }

def display_progress(stats):
    """显示进度"""
    if not stats:
        print("No progress data available")
        return

    total = stats['total_categories']
    completed = stats['completed_count']
    percent = (completed / total * 100) if total > 0 else 0

    # 清屏（Windows/Unix兼容）
    os.system('cls' if os.name == 'nt' else 'clear')

    print('=' * 80)
    print('                234 Category NER Processing - Real-Time Monitor')
    print('=' * 80)
    print()

    # 总进度
    print(f'Total Progress: {completed}/{total} categories ({percent:.1f}%)')

    # 进度条
    bar_width = 70
    filled = int(bar_width * completed / total) if total > 0 else 0
    bar = '#' * filled + '-' * (bar_width - filled)
    print(f'[{bar}]')
    print()

    # 状态统计
    print(f'Completed: {completed:3d}   Failed: {stats["failed_count"]:3d}   Remaining: {stats["remaining"]:3d}')
    print(f'Total Items Processed: {stats["total_items"]:,} records')
    print()

    # 按组别统计
    print('-' * 80)
    print('Category Groups Breakdown:')
    print('-' * 80)

    group_names = {
        '食品饮料类': 'Food & Beverage',
        '日用洗护类': 'Personal Care',
        '母婴用品类': 'Mother & Baby',
        '服装鞋帽类': 'Apparel',
        '电子配件类': 'Electronics',
        '家居用品类': 'Home & Kitchen',
        '美妆护肤类': 'Beauty Care'
    }

    for cn_name, en_name in group_names.items():
        count = stats['groups_stats'].get(cn_name, 0)
        if count > 0:
            bar_len = 30
            filled_len = min(int(bar_len * count / 50), bar_len)  # 假设每组最多50个
            mini_bar = '#' * filled_len + '-' * (bar_len - filled_len)
            print(f'{en_name:20s} [{mini_bar}] {count:3d} categories')

    print()

    # 最近完成
    if stats['recently_completed']:
        print('Recently Completed (Last 5):')
        for cat in stats['recently_completed']:
            print(f'  - {cat}')

    print()
    print('=' * 80)
    print(f'Last Updated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('Press Ctrl+C to exit...')

if __name__ == '__main__':
    try:
        while True:
            stats = get_progress_stats()
            display_progress(stats)
            time.sleep(5)  # 每5秒刷新一次
    except KeyboardInterrupt:
        print('\nMonitoring stopped.')
