#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
234品类批量NER识别 - DeepSeek V3
模型: Pro/deepseek-ai/DeepSeek-V3 (SiliconFlow)

功能：
1. 自动扫描 category_csv/ 目录下所有CSV文件
2. 根据品类名称自动分组
3. 动态构建针对性提示词（含Few-Shot）
4. 异步并发处理（15并发）
5. 结果验证与标记
6. 断点续跑支持
"""

import csv
import json
import os
import sys
import asyncio
import aiohttp
import time
import random
import re
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# 导入配置
from category_config import (
    CATEGORY_GROUPS, FEW_SHOT_EXAMPLES, FIELD_MAP, OUTPUT_HEADER
)


# ==================== API 配置 ====================
MODEL_NAME = "Pro/deepseek-ai/DeepSeek-V3"
API_KEY = "sk-rxtfeeajhzggovvqowryvuresvyceokafnsfkuvfrojabyot"
API_BASE = "https://api.siliconflow.cn/v1"
CHAT_ENDPOINT = f"{API_BASE}/chat/completions"

# ==================== 运行配置 ====================
CONCURRENCY = 15  # 并发请求数
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "category_csv"
OUTPUT_DIR = BASE_DIR / "category_output"
CACHE_DIR = OUTPUT_DIR / "cache"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"


# ==================== 进度监控 ====================
class ProgressMonitor:
    def __init__(self, total: int, category_name: str):
        self.total = total
        self.processed = 0
        self.start_time = time.time()
        self.category_name = category_name
        self.lock = asyncio.Lock()

    async def update(self, count: int = 1):
        async with self.lock:
            self.processed += count
            elapsed = time.time() - self.start_time
            speed = self.processed / elapsed if elapsed > 0 else 0
            percent = self.processed / self.total * 100 if self.total > 0 else 0
            remaining = (self.total - self.processed) / speed if speed > 0 else 0

            if self.processed % 10 == 0 or self.processed == self.total:
                print(f"  [{self.category_name}] {percent:.1f}% ({self.processed}/{self.total}) | "
                      f"Speed: {speed:.2f}it/s | ETA: {remaining:.1f}s")


# ==================== 品类分类器 ====================
def classify_category(category_name: str) -> str:
    """根据品类名称返回组别ID"""
    # 精确匹配
    for group_id, group_info in CATEGORY_GROUPS.items():
        if category_name in group_info["categories"]:
            return group_id

    # 模糊匹配（关键字）
    keyword_map = {
        "food_beverage": ["米", "油", "酒", "饮", "茶", "奶", "糖", "食", "味", "蛋", "肉", "菜", "果", "冻", "罐"],
        "personal_care": ["洗", "膏", "护", "皂", "纸", "巾", "卫生", "清洁", "杀虫"],
        "mother_baby": ["婴", "幼", "孕", "妇", "奶粉", "尿", "辅食", "保健", "营养"],
        "apparel": ["衣", "服", "鞋", "袜", "裤", "帽", "带", "饰", "手套"],
        "electronics": ["电", "充", "配件", "数码", "机"],
        "beauty_care": ["保养", "化妆", "美容"],
        "home_kitchen": ["杯", "壶", "具", "锅", "碗", "盘", "刀", "架", "盆", "桶", "办公", "文", "玩"],
    }

    for group_id, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in category_name:
                return group_id

    # 默认归类
    return "home_kitchen"


# ==================== 动态提示词构建 ====================
def build_prompt(product_name: str, barcode: str, category: str, group_id: str) -> str:
    """根据组别动态构建提示词"""
    group = CATEGORY_GROUPS[group_id]
    examples = FEW_SHOT_EXAMPLES.get(group_id, [])

    # 生成字段规则说明
    emphasis_text = ""
    if group.get("field_emphasis"):
        fields = "、".join(group["field_emphasis"])
        emphasis_text = f"\n**重点关注字段**: {fields}"

    suppress_text = ""
    if group.get("field_suppress"):
        fields = "、".join(group["field_suppress"])
        suppress_text = f"\n**禁用字段（必须留空）**: {fields}"

    instructions_text = ""
    if group.get("field_instructions"):
        instructions_text = "\n**字段使用说明**:\n"
        for field, instruction in group["field_instructions"].items():
            instructions_text += f"   - {field}: {instruction}\n"

    # 格式化 Few-Shot 示例
    examples_text = ""
    for i, example in enumerate(examples[:3], 1):
        output_str = json.dumps(example["output"], ensure_ascii=False, indent=6)
        examples_text += f"""
{i}. 「{example["product_name"]}」
   {output_str}
"""

    prompt = f"""你是一个商品数据治理专家。请对以下商品标题进行NER识别，严格提取关键属性。

商品标题：{product_name}
商品品类：{category}
所属分组：{group["name"]}

核心原则（必须严格遵守）：
1. **全覆盖原则**：名称中所有具有语义的实体词汇都必须被识别并填入对应的标签中。
2. **一词一用（严禁重复）**：同一个词只能归属到一个实体标签中，禁止在多个字段中重复出现。
3. **保持完整语义**：提取时应保持词汇的语义完整性，避免拆分。
4. **原文提取**：所有提取的内容必须是标题中原有的词汇，禁止无中生有。
5. **品类适配**：根据该品类特性使用合适的字段。

**品牌粘连处理规则（重要）**：
- 品牌通常位于标题开头
- 当品牌后紧跟其他属性词时，必须正确拆分，不能将属性词误识别为品牌的一部分
- 常见品牌粘连情况示例：
  - "金龙鱼压榨一级花生油" → 品牌"金龙鱼"，加工工艺"压榨"，等级"一级"
  - "海天金标生抽" → 品牌"海天"，子品牌/系列"金标"
  - "云南白药金口健牙膏" → 品牌"云南白药"，子品牌/系列"金口健"
  - "鲁花5S压榨花生油" → 品牌"鲁花"，认证/标准"5S"，加工工艺"压榨"
{emphasis_text}
{suppress_text}
{instructions_text}

字段定义与示例：
   - **[品牌]**：品牌名称（优先识别，注意与后续词拆分）
   - **[子品牌/系列]**：产品系列名
   - **[品类]**：核心产品形态
   - **[口味1/2/3]**：仅食品类使用，如浓香茄汁味、海苔味、烧烤口味
   - **[加工工艺]**：如压榨、浸出、发酵等
   - **[认证/标准]**：如有机认证、ISO 9001、3C
   - **[特性1/2/3]**：香型、成分特点，如清凉、木糖醇、无糖
   - **[功能1/2/3]**：主要功效，如去屑、柔顺、控油
   - **[设计类型]**：如换头装、折叠式、伸缩式、分体式、磁吸式、卡扣式、插拔式
   - **[套装/组合]**：如礼盒、套装、组合装
   - **[包装类型]**：如罐装、听装（注意：单字"袋"、"盒"是包装单位）
   - **[包装材质]**：如玻璃瓶、铝罐、可降解纸盒
   - **[包装数量]**：如*12中的数字部分
   - **[包装单位]**：如桶、瓶、支、盒、袋
   - **[规格]**：重量/体积，如12g、5L、500g
   - **[容量]**：存储/电池容量，如1TB、300mAh（容器、电池、存储设备专用）
   - **[尺寸]**：尺寸规格，如160mm*200mm、L、XL
   - **[颜色]**：如雅川青、黑色
   - **[材质]**：产品主体材质，如纯棉、不锈钢
   - **[产地]**：产地信息
   - **[等级]**：如一级、特级
   - **[纯度]**：如100%、53度、53%vol、纯、纯正
   - **[适用人群/对象]**：如婴幼儿、孕妇、男性专用、老人
   - **[场景]**：如家庭装、旅行装、餐饮装
   - **[型号]**：如FS928

参考示例（Few-Shot）：
{examples_text}

请返回单个JSON对象：
{{
    "id": "{barcode}",
    "企业": "", "品牌": "", "子品牌/系列": "", "品类": "",
    "口味1": "", "口味2": "", "口味3": "",
    "加工工艺": "", "认证/标准": "",
    "特性1": "", "特性2": "", "特性3": "",
    "功能1": "", "功能2": "", "功能3": "",
    "设计类型": "", "套装/组合": "", "包装类型": "", "包装材质": "",
    "包装数量": "", "包装单位": "",
    "规格": "", "容量": "", "尺寸": "", "颜色": "", "材质": "",
    "产地": "", "等级": "", "纯度": "", "适用人群/对象": "", "场景": "", "型号": ""
}}
"""
    return prompt


# ==================== 结果验证 ====================
def validate_result(result: Dict, product_name: str, group_id: str) -> List[str]:
    """验证结果，返回警告列表"""
    warnings = []
    group = CATEGORY_GROUPS[group_id]

    # 1. 字段值重复检查
    value_map = defaultdict(list)
    for field, value in result.items():
        if field in ['id', 'barcode', 'product_name', 'category_input']:
            continue
        if value and str(value).strip():
            value_map[str(value).strip()].append(field)

    for value, fields in value_map.items():
        if len(fields) > 1 and len(value) > 1:  # 忽略单字符重复
            warnings.append(f"值重复: '{value}' 出现在: {', '.join(fields)}")

    # 2. 规格格式检查
    spec = result.get('规格', '')
    if spec and not re.search(r'\d+\s*(g|kg|ml|L|斤|两|升|毫升|克|千克)', spec, re.IGNORECASE):
        if not re.search(r'\d+', spec):  # 如果连数字都没有
            warnings.append(f"规格格式可疑: '{spec}'")

    # 3. 禁用字段检查
    for field in group.get("field_suppress", []):
        if result.get(field):
            warnings.append(f"禁用字段被使用: {field}='{result.get(field)}'")

    # 4. 来源验证（检查提取值是否在原标题中）
    for field, value in result.items():
        if field in ['id', 'barcode', 'product_name', 'category_input']:
            continue
        if value and str(value).strip():
            # 允许部分匹配（因为可能有分词）
            val = str(value).strip()
            if len(val) > 2 and val not in product_name:
                # 尝试去除常见后缀
                val_clean = val.rstrip('味型装色')
                if val_clean and val_clean not in product_name:
                    warnings.append(f"提取值可能不在原标题: {field}='{value}'")

    return warnings


# ==================== 缓存管理 ====================
def get_cache_key(item: Dict, category: str) -> str:
    b = str(item.get('barcode', '')).strip()
    n = str(item.get('product_name', '')).strip()
    if not n:
        n = str(item.get('name', '')).strip()
    return f"{b}_{n}_{category}"


def load_cache(cache_file: Path) -> Dict[str, Any]:
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"  缓存文件损坏，重新开始: {cache_file}")
            return {}
    return {}


def save_cache(cache: Dict[str, Any], cache_file: Path):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ==================== 进度管理（断点续跑） ====================
def load_progress() -> Dict:
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"completed": [], "failed": {}}
    return {"completed": [], "failed": {}}


def save_progress(progress: Dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ==================== 核心处理逻辑 ====================
async def process_item(
    session: aiohttp.ClientSession,
    item: Dict,
    cache: Dict,
    monitor: ProgressMonitor,
    category: str,
    group_id: str
) -> Dict:
    """处理单条数据"""
    key = get_cache_key(item, category)

    # 检查缓存
    if key in cache:
        await monitor.update()
        return {**item, **cache[key], "_from_cache": True}

    product_name = str(item.get('product_name', '')).strip()
    if not product_name:
        product_name = str(item.get('name', '')).strip()

    if not product_name:
        await monitor.update()
        return item

    # 构造Prompt
    prompt = build_prompt(product_name, item.get('barcode', ''), category, group_id)

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    retry_count = 0
    max_retries = 5

    while retry_count < max_retries:
        try:
            async with session.post(CHAT_ENDPOINT, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    try:
                        parsed = json.loads(content)
                        # 验证结果
                        warnings = validate_result(parsed, product_name, group_id)
                        parsed["_warnings"] = warnings
                        # 写入缓存
                        cache[key] = parsed
                        await monitor.update()
                        return {**item, **parsed}
                    except json.JSONDecodeError:
                        print(f"  JSON解析失败: {content[:100]}...")
                        break
                elif response.status == 429:
                    retry_count += 1
                    wait_time = random.uniform(1, 3) * retry_count
                    print(f"  Rate limit, retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  API Error {response.status}")
                    break
        except Exception as e:
            print(f"  Request failed: {e}")
            retry_count += 1
            await asyncio.sleep(1 * retry_count)

    await monitor.update()
    return item


async def process_category(
    input_file: Path,
    output_file: Path,
    cache_file: Path,
    category: str,
    group_id: str,
    group_name: str
) -> Dict:
    """处理单个品类"""
    # 读取输入
    items = []
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        items = list(reader)

    if not items:
        print(f"  [跳过] {category}: 无数据")
        return {"status": "empty", "count": 0}

    print(f"  [处理] {category} ({len(items)} 条) -> {group_name}")

    # 加载缓存
    cache = load_cache(cache_file)
    cached_count = sum(1 for item in items if get_cache_key(item, category) in cache)
    if cached_count > 0:
        print(f"  [缓存] 已有 {cached_count}/{len(items)} 条缓存")

    monitor = ProgressMonitor(len(items), category)

    # 并发处理
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_item(session, item, cache, monitor, category, group_id)
            for item in items
        ]
        results = await asyncio.gather(*tasks)

    # 保存缓存
    save_cache(cache, cache_file)

    # 统计验证结果
    warning_count = sum(1 for r in results if r.get("_warnings"))

    # 写入结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        for res in results:
            row = {
                "barcode": res.get('barcode', ''),
                "product_name": res.get('product_name', '') or res.get('name', ''),
                "category_input": category,
                "group_name": group_name
            }
            for k, v in FIELD_MAP.items():
                row[v] = res.get(k, '')
            writer.writerow(row)

    print(f"  [完成] {category}: {len(items)} 条, 警告 {warning_count} 条")

    return {
        "status": "success",
        "count": len(items),
        "warnings": warning_count
    }


# ==================== 批量处理入口 ====================
async def process_all_categories():
    """批量处理所有品类"""
    print("=" * 60)
    print("234 品类批量 NER 处理")
    print("=" * 60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 扫描所有 CSV 文件
    csv_files = list(INPUT_DIR.glob("*.csv"))
    print(f"\n发现 {len(csv_files)} 个品类文件")

    # 加载进度
    progress = load_progress()
    completed = set(progress.get("completed", []))

    # 统计
    total_categories = len(csv_files)
    processed_categories = 0
    total_items = 0
    total_warnings = 0

    # 按组别整理
    groups_stats = defaultdict(lambda: {"count": 0, "items": 0})

    start_time = time.time()

    for csv_file in csv_files:
        category_name = csv_file.stem

        # 跳过已完成
        if category_name in completed:
            print(f"\n[跳过] {category_name} (已完成)")
            processed_categories += 1
            continue

        # 自动分组
        group_id = classify_category(category_name)
        group_name = CATEGORY_GROUPS[group_id]["name"]

        # 构建路径
        output_file = OUTPUT_DIR / group_name / f"{category_name}_result.csv"
        cache_file = CACHE_DIR / f"{category_name}_cache.json"

        try:
            result = await process_category(
                csv_file, output_file, cache_file,
                category_name, group_id, group_name
            )

            if result["status"] == "success":
                total_items += result["count"]
                total_warnings += result["warnings"]
                groups_stats[group_name]["count"] += 1
                groups_stats[group_name]["items"] += result["count"]

            # 标记完成
            progress["completed"].append(category_name)
            save_progress(progress)
            processed_categories += 1

        except Exception as e:
            print(f"\n[错误] {category_name}: {e}")
            progress["failed"][category_name] = str(e)
            save_progress(progress)

    # 汇总报告
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("处理完成！汇总报告：")
    print("=" * 60)
    print(f"品类数量: {processed_categories}/{total_categories}")
    print(f"数据总量: {total_items} 条")
    print(f"警告数量: {total_warnings} 条")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"\n按组别统计:")
    for group_name, stats in sorted(groups_stats.items()):
        print(f"  - {group_name}: {stats['count']} 个品类, {stats['items']} 条数据")
    print(f"\n输出目录: {OUTPUT_DIR}")


# ==================== 单品类处理（调试用） ====================
async def process_single_category(category_name: str):
    """处理单个品类（调试用）"""
    csv_file = INPUT_DIR / f"{category_name}.csv"
    if not csv_file.exists():
        print(f"文件不存在: {csv_file}")
        return

    group_id = classify_category(category_name)
    group_name = CATEGORY_GROUPS[group_id]["name"]

    output_file = OUTPUT_DIR / group_name / f"{category_name}_result.csv"
    cache_file = CACHE_DIR / f"{category_name}_cache.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    await process_category(
        csv_file, output_file, cache_file,
        category_name, group_id, group_name
    )


# ==================== 主入口 ====================
if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 支持单品类处理（调试）
    if len(sys.argv) > 1:
        category = sys.argv[1]
        print(f"单品类处理: {category}")
        asyncio.run(process_single_category(category))
    else:
        # 批量处理所有品类
        asyncio.run(process_all_categories())
