#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用商品NER识别 - DeepSeek V3 优化版 (高并发单条模式)
模型: Pro/deepseek-ai/DeepSeek-V3 (SiliconFlow)
适用于多品类商品属性提取
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
from typing import Dict, List, Any


MODEL_NAME = "Pro/deepseek-ai/DeepSeek-V3"
API_KEY = "sk-rxtfeeajhzggovvqowryvuresvyceokafnsfkuvfrojabyot"
API_BASE = "https://api.siliconflow.cn/v1"
CHAT_ENDPOINT = f"{API_BASE}/chat/completions"

# 运行配置
CONCURRENCY = 15      # 并发请求数
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "category_input.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "ner_results_category_v1.csv")
CACHE_FILE = os.path.join(BASE_DIR, "ner_cache_category_v1.json")

# 支持命令行参数
if len(sys.argv) > 3:
    INPUT_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]
    CACHE_FILE = sys.argv[3]
    print(f"Using command line args:\nInput: {INPUT_FILE}\nOutput: {OUTPUT_FILE}\nCache: {CACHE_FILE}")


# 可视化打印
class ProgressMonitor:
    def __init__(self, total):
        self.total = total
        self.processed = 0
        self.start_time = time.time()
        self.lock = asyncio.Lock()

    async def update(self, count=1):
        async with self.lock:
            self.processed += count
            elapsed = time.time() - self.start_time
            speed = self.processed / elapsed if elapsed > 0 else 0
            percent = self.processed / self.total * 100 if self.total > 0 else 0
            remaining = (self.total - self.processed) / speed if speed > 0 else 0
            if self.processed % 5 == 0 or self.processed == self.total:
                print(f"[Progress] {percent:.1f}% ({self.processed}/{self.total}) | Time: {elapsed:.1f}s | Speed: {speed:.2f}it/s | ETA: {remaining:.1f}s")

# 通用提示词模板（适配多品类）
PROMPT_TEMPLATE = """你是一个商品数据治理专家。请对以下商品标题进行NER识别，严格提取关键属性。

商品标题：{product_name}
商品品类：{category}

核心原则（必须严格遵守）：
1. **全覆盖原则（宁滥勿缺）**：
   - 名称中**所有**具有语义的实体词汇都**必须**被识别并填入对应的标签中。
   - **严禁**出现"觉得不合适就不填"的情况。

2. **一词一用（严禁重复）**：
   - 同一个词（或字符片段）只能归属到一个实体标签中，禁止在多个字段中重复出现。
   - **特别注意**：如果"袋"、"盒"、"桶"等词已经被提取为 **[包装单位]**，则 **绝不能** 再填入 **[包装类型]**。

3. **保持完整语义（避免拆分）**：
   - 提取时应保持词汇的语义完整性，避免将一个完整的概念拆分成无关的碎片。

4. **原文提取**：
   - 所有提取的内容必须是标题中原有的词汇，禁止无中生有。

5. **品类适配原则**：
   - 根据商品品类特性，灵活运用字段。
   - 食品类：优先使用[口味]、[加工工艺]、[认证/标准]字段。
   - 日用品类（如牙膏、洗护）：[口味]留空，相关描述填入[特性]。
   - 电子产品：优先使用[容量]、[型号]、[材质]、[尺寸]字段。
   - 服装类：优先使用[颜色]、[材质]、[尺寸]、[设计类型]字段。

字段定义与优先级：
   - **[企业]**：生产企业名称（如有明确标注）。
   - **[品牌]**：品牌名称（优先识别）。
   - **[子品牌/系列]**：产品系列名、子品牌。
   - **[品类]**：核心产品形态（如牙膏、食用油、手机、T恤等）。

   - **[口味1]** / **[口味2]** / **[口味3]**：
     - **仅适用于食品类商品**。
     - 常见：草莓味、巧克力味、原味、香辣、麻辣、清淡等。
     - **非食品类商品此字段留空**，相关描述填入[特性]。

   - **[加工工艺]**：明确的工艺描述。
     - 食品类：压榨、浸出、古法、物理压榨、冷压、发酵、烘焙等。
     - 其他类：手工、机织、注塑、冲压等。

   - **[认证/标准]**：认证/标准信息。
     - 如有机认证、ISO 9001、FDA认证、国标GB、欧标CE等。

   - **[特性1]** / **[特性2]** / **[特性3]**：（如有多个特性，依次填入）
     - 产品的香型、口感、成分特点或其他描述性特征。
     - 食品类：木糖醇、无糖、低脂、高蛋白、富含维生素等。
     - 日用品类：薄荷、留兰香、绿茶、清凉、小苏打、益生菌等。
     - 电子产品：智能、快充、无线、防水、高清等。
     - 服装类：透气、速干、修身、宽松、复古等。

   - **[功能1]** / **[功能2]** / **[功能3]**：（如有多个功能，依次填入）
     - 产品的主要功能或功效。
     - 日用品：美白、抗过敏、防蛀、固齿、护龈、清新口气、去屑、柔顺等。
     - 电子产品：导航、拍照、游戏、办公等。
     - 食品：补钙、补铁、提神、助消化等。

   - **[设计类型]**：设计风格或结构类型。
     - 如换头装、折叠式、伸缩式、分体式、磁吸式、卡扣式、插拔式、圆领、V领、直筒、修身等。

   - **[套装/组合]**：套装/组合信息。
     - 如礼盒、套装、组合装、家庭装、分享装等。

   - **[包装类型]**：包装形式描述。
     - 如支装、盒装、瓶装、罐装、听装、袋装、桶装。
     - **注意**：单字"袋"、"盒"、"支"通常是[包装单位]，**严禁**填入此字段。

   - **[包装材质]**：包装材料。
     - 如玻璃瓶、铝罐、可降解纸盒、塑料袋、牛皮纸袋等。

   - **[包装数量]**：包装数量数字。
     - 如遇到*2、x3、*12等，只提取数字（如2、3、12）。

   - **[包装单位]**：包装计量单位。
     - 如支、盒、套、包、袋、桶、瓶、罐、听、条、个等。

   - **[规格]**：产品重量或体积规格。
     - 如120g、140g、90g、5L、500ml、1kg等。

   - **[容量]**：存储或电池容量（主要用于电子产品）。
     - 如1TB、256GB、5000mAh、3000mAh等。

   - **[尺寸]**：产品尺寸规格。
     - 如160mm*200mm、L、XL、M、42码、39码等。

   - **[颜色]**：颜色描述。
     - 如黑色、白色、红色、雅川青、星空灰、薄荷绿等。

   - **[材质]**：产品主体材质。
     - 如塑料、不锈钢、纯棉、涤纶、真皮、硅胶、陶瓷等。

   - **[产地]**：产地信息。
     - 如中国、日本、法国、山东、云南等。

   - **[等级]**：产品等级。
     - 如一级、特级、优等品、A级等。

   - **[纯度]**：纯度描述。
     - 如100%、纯、纯正、纯天然等。

   - **[适用人群/对象]**：目标用户群体。
     - 如儿童、成人、孕妇、老人、男士、女士等。

   - **[场景]**：使用场景。
     - 如家庭装、旅行装、便携、户外、办公、运动等。

   - **[型号]**：产品型号。
     - 如FS928、iPhone 15、Model X等。

参考示例（Few-Shot）：
1. 食品类示例：「金龙鱼压榨一级花生油5L桶装」
   {{
       "id": "{barcode}",
       "企业": "", "品牌": "金龙鱼", "子品牌/系列": "", "品类": "花生油",
       "口味1": "", "口味2": "", "口味3": "",
       "加工工艺": "压榨", "认证/标准": "",
       "特性1": "", "特性2": "", "特性3": "",
       "功能1": "", "功能2": "", "功能3": "",
       "设计类型": "", "套装/组合": "", "包装类型": "桶装", "包装材质": "", "包装数量": "", "包装单位": "",
       "规格": "5L", "容量": "", "尺寸": "", "颜色": "", "材质": "", "产地": "", "等级": "一级", "纯度": "", "适用人群/对象": "", "场景": "", "型号": ""
   }}

2. 日用品类示例：「云南白药金口健牙膏益优清新冰柠薄荷型105g」
   {{
       "id": "{barcode}",
       "企业": "", "品牌": "云南白药", "子品牌/系列": "金口健", "品类": "牙膏",
       "口味1": "", "口味2": "", "口味3": "",
       "加工工艺": "", "认证/标准": "",
       "特性1": "益优清新", "特性2": "冰柠薄荷型", "特性3": "",
       "功能1": "", "功能2": "", "功能3": "",
       "设计类型": "", "套装/组合": "", "包装类型": "", "包装材质": "", "包装数量": "", "包装单位": "",
       "规格": "105g", "容量": "", "尺寸": "", "颜色": "", "材质": "", "产地": "", "等级": "", "纯度": "", "适用人群/对象": "", "场景": "", "型号": ""
   }}

3. 电子产品示例：「小米13 Pro 黑色 12GB+256GB 5G智能手机」
   {{
       "id": "{barcode}",
       "企业": "", "品牌": "小米", "子品牌/系列": "", "品类": "智能手机",
       "口味1": "", "口味2": "", "口味3": "",
       "加工工艺": "", "认证/标准": "",
       "特性1": "5G", "特性2": "", "特性3": "",
       "功能1": "", "功能2": "", "功能3": "",
       "设计类型": "", "套装/组合": "", "包装类型": "", "包装材质": "", "包装数量": "", "包装单位": "",
       "规格": "", "容量": "256GB", "尺寸": "12GB", "颜色": "黑色", "材质": "", "产地": "", "等级": "", "纯度": "", "适用人群/对象": "", "场景": "", "型号": "小米13 Pro"
   }}

请返回单个JSON对象：
{{
    "id": "{barcode}",
    "企业": "", "品牌": "", "子品牌/系列": "", "品类": "",
    "口味1": "", "口味2": "", "口味3": "",
    "加工工艺": "", "认证/标准": "",
    "特性1": "", "特性2": "", "特性3": "",
    "功能1": "", "功能2": "", "功能3": "",
    "设计类型": "", "套装/组合": "", "包装类型": "", "包装材质": "", "包装数量": "", "包装单位": "",
    "规格": "", "容量": "", "尺寸": "", "颜色": "", "材质": "",
    "产地": "", "等级": "", "纯度": "", "适用人群/对象": "", "场景": "", "型号": ""
}}
"""

# 字段映射
FIELD_MAP = {
    "企业": "enterprise",
    "品牌": "brand",
    "子品牌/系列": "sub_brand",
    "品类": "category",
    "口味1": "flavor_1",
    "口味2": "flavor_2",
    "口味3": "flavor_3",
    "加工工艺": "process",
    "认证/标准": "certification",
    "特性1": "attribute_1",
    "特性2": "attribute_2",
    "特性3": "attribute_3",
    "功能1": "function_1",
    "功能2": "function_2",
    "功能3": "function_3",
    "设计类型": "design_type",
    "套装/组合": "bundle_type",
    "包装类型": "pack_type",
    "包装材质": "pack_material",
    "包装数量": "pack_quantity",
    "包装单位": "pack_unit",
    "规格": "spec",
    "容量": "capacity",
    "尺寸": "size",
    "颜色": "color",
    "材质": "material",
    "产地": "origin",
    "等级": "grade",
    "纯度": "purity",
    "适用人群/对象": "target_user",
    "场景": "scene",
    "型号": "model"
}

OUTPUT_HEADER = ["barcode", "product_name", "category_input"] + list(FIELD_MAP.values())

# --- 缓存管理 ---
def get_cache_key(item: Dict) -> str:
    b = str(item.get('barcode', '')).strip()
    n = str(item.get('product_name', '')).strip()
    if not n:
        n = str(item.get('name', '')).strip()
    c = str(item.get('category', '')).strip()
    return f"{b}_{n}_{c}"

def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("缓存文件损坏，重新开始...")
            return {}
    return {}

def save_cache(cache: Dict[str, Any]):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# 核心处理逻辑
async def process_item(session: aiohttp.ClientSession, item: Dict, cache: Dict, monitor: ProgressMonitor):
    key = get_cache_key(item)

    # 检查缓存
    if key in cache:
        await monitor.update()
        return {**item, **cache[key]}

    product_name = str(item.get('product_name', '')).strip()
    if not product_name:
        product_name = str(item.get('name', '')).strip()

    category = str(item.get('category', '')).strip()
    if not category:
        category = "未知品类"

    if not product_name:
        await monitor.update()
        return item

    # 构造Prompt（加入品类信息）
    prompt = PROMPT_TEMPLATE.format(
        product_name=product_name,
        barcode=item.get('barcode', ''),
        category=category
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
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
                        # 结果写入缓存
                        cache[key] = parsed
                        await monitor.update()
                        return {**item, **parsed}
                    except json.JSONDecodeError:
                        print(f"JSON解析失败: {content}")
                        break
                elif response.status == 429:
                    retry_count += 1
                    wait_time = random.uniform(1, 3) * retry_count
                    print(f"Rate limit hit, retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"API Error {response.status}: {await response.text()}")
                    break
        except Exception as e:
            print(f"Request failed: {e}")
            retry_count += 1
            await asyncio.sleep(1 * retry_count)

    await monitor.update()
    return item

async def main():
    # 读取输入
    items = []
    with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        items = list(reader)

    print(f"共加载 {len(items)} 条数据")

    # 加载缓存
    cache = load_cache()
    monitor = ProgressMonitor(len(items))

    # 并发处理
    print("Starting concurrent processing...")
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [process_item(session, item, cache, monitor) for item in items]
            results = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Processing complete.")

    # 保存缓存
    save_cache(cache)

    print("Writing results to", OUTPUT_FILE)
    print("Header:", OUTPUT_HEADER)

    # 写入结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        for res in results:
            # 展平结果
            row = {
                "barcode": res.get('barcode', ''),
                "product_name": res.get('product_name', '') or res.get('name', ''),
                "category_input": res.get('category', '')
            }
            # 填充提取字段
            for k, v in FIELD_MAP.items():
                row[v] = res.get(k, '')
            writer.writerow(row)

    print(f"\n完成! 结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
