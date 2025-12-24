#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙膏NER识别 - DeepSeek V3 优化版 (高并发单条模式)
模型: Pro/deepseek-ai/DeepSeek-V3 (SiliconFlow)
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
CONCURRENCY = 15      # 并发请求数（不建议再调高了）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "ner_results_牙膏.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "ner_results_toothpaste_full_v4.csv")
CACHE_FILE = os.path.join(BASE_DIR, "ner_cache_toothpaste_full_v4.json")

#这里在运行中可能会出现request failed 情况，无需操作，等待重连即可，因为我这边使用15并发就会有点频繁访问api，后续有缓存和重试机制，不会造成一个丢失情况


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

#提示词
PROMPT_TEMPLATE = """你是一个商品数据治理专家。请对以下【牙膏】类商品标题进行NER识别，严格提取关键属性。

商品标题：{product_name}

核心原则（必须严格遵守）：
1. **全覆盖原则（宁滥勿缺）**：
   - 名称中**所有**具有语义的实体词汇都**必须**被识别并填入对应的标签中。
   - **严禁**出现“觉得不合适就不填”的情况。


2. **一词一用（严禁重复）**：
   - 同一个词（或字符片段）只能归属到一个实体标签中，禁止在多个字段中重复出现。
   - **特别注意**：如果“袋”、“盒”、“桶”等词已经被提取为 **[包装单位]**，则 **绝不能** 再填入 **[包装类型]**。

3. **保持完整语义（避免拆分）**：
   - 提取时应保持词汇的语义完整性，避免将一个完整的概念（如“益优清新”）拆分成无关的碎片。

4. **原文提取**：
   - 所有提取的内容必须是标题中原有的词汇，禁止无中生有。

字段定义与优先级：
   - **[品牌]**：品牌名称（优先识别）。
     - 常见：云南白药、高露洁、佳洁士、黑人、中华、冷酸灵、两面针、舒适达、舒客、纳爱斯、狮王、花王、皓乐齿、片仔癀、六必治、小巨蛋、参半。
   - **[子品牌/系列]**：产品系列名。
     - 常见：金口健、朗健、益优清新、360、全优七效、多效护理、抗敏感、酵素、白酵素、专研、儿童、爱牙牙、植雅。
   - **[品类]**：核心产品形态。
     - 常见：牙膏
   - **[口味1]** / **[口味2]** / **[口味3]**：
     - **严禁填入**。牙膏属于非食品类，没有“口味”属性。所有类似“薄荷”、“水果味”等描述，**必须**填入 **[特性]** 字段。
   - **[加工工艺]**：明确的工艺描述，以食用油类别举例:如"压榨"、"浸出"、"古法"、"物理压榨"等，如非明确，留空。
   - **[认证/标准]**：认证/标准，如有机认证、ISO 9001、FDA认证
   - **[特性1]** / **[特性2]** / **[特性3]**：(如有多个特性，依次填入)
     - 包含产品的香型、口感或成分特点。
     - 常见：薄荷、留兰香、绿茶、柠檬、草莓、水果、海盐、茉莉、茶香、激爽薄荷、冰柠薄荷、香甜草莓。
     - 其他特点：清凉、木糖醇、无糖、酵素、益生菌、小苏打、竹炭。
   - **[功能1]** / **[功能2]** / **[功能3]**：(如有多个功能，依次填入)
     - 如去屑、柔顺、美白、抗过敏、防蛀、固齿、护龈、抗敏、清新口气、去牙菌斑、多效。
   - **[设计类型]**：设计类型，如换头装、折叠式、伸缩式、分体式、磁吸式、卡扣式、插拔式
   - **[套装/组合]**：套装/组合，如礼盒、套装、组合装
   - **[包装类型]**：如支装、盒装、家庭装、罐装、听装。
     - **注意**：单字“袋”、“盒”、“支”通常是[包装单位]，**严禁**填入此字段。
   - **[包装材质]**：如“玻璃瓶”、“铝罐”、“可降解纸盒”。
   - **[包装数量]**：如遇到*2、x3、*12等只提取相应的数字即可（表示包装数量）。
   - **[包装单位]**：如支、盒、套、包、袋、桶、瓶等。
   - **[规格]**：如120g、140g、90g、12g、5L、500g等（重量/体积）。
   - **[容量]**：如1TB, 300mAh（牙膏通常留空）。
   - **[尺寸]**：如160mm*200mm（牙膏通常留空）。
   - **[颜色]**：如雅川青（牙膏通常留空）。
   - **[材质]**：如塑料、不锈钢（牙膏通常留空）。
   - **[产地]**：产地。
   - **[等级]**：如一级等。
   - **[纯度]**：如100%、纯、纯正等。
   - **[适用人群/对象]**：如儿童、成人、孕妇等。
   - **[场景]**：如家庭装、旅行装。
   - **[型号]**：如FS928。

参考示例（Few-Shot）：
1. 「云南白药金口健牙膏益优清新冰柠薄荷型105g」
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

2. 「奥冰小苏打祛渍亮白双重薄荷牙膏180g*2」
   {{
       "id": "{barcode}",
       "企业": "", "品牌": "奥冰", "子品牌/系列": "", "品类": "牙膏", 
       "口味1": "", "口味2": "", "口味3": "", 
       "加工工艺": "", "认证/标准": "", 
       "特性1": "小苏打", "特性2": "双重薄荷", "特性3": "", 
       "功能1": "祛渍", "功能2": "亮白", "功能3": "", 
       "设计类型": "", "套装/组合": "", "包装类型": "", "包装材质": "", "包装数量": "2", "包装单位": "", 
       "规格": "180g", "容量": "", "尺寸": "", "颜色": "", "材质": "", "产地": "", "等级": "", "纯度": "", "适用人群/对象": "", "场景": "", "型号": ""
   }}

3. 「两面针爱牙牙儿童健牙膏香甜草莓香50g」
   {{
       "id": "{barcode}",
       "企业": "", "品牌": "两面针", "子品牌/系列": "爱牙牙", "品类": "健牙膏", 
       "口味1": "", "口味2": "", "口味3": "", 
       "加工工艺": "", "认证/标准": "", 
       "特性1": "香甜", "特性2": "草莓香", "特性3": "", 
       "功能1": "", "功能2": "", "功能3": "", 
       "设计类型": "", "套装/组合": "", "包装类型": "", "包装材质": "", "包装数量": "", "包装单位": "", 
       "规格": "50g", "容量": "", "尺寸": "", "颜色": "", "材质": "", "产地": "", "等级": "", "纯度": "", "适用人群/对象": "儿童", "场景": "", "型号": ""
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
    return f"{b}_{n}"

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
        
    if not product_name:
        await monitor.update()
        return item

    # 构造Prompt
    prompt = PROMPT_TEMPLATE.format(product_name=product_name, barcode=item.get('barcode', ''))
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0, #这里为了防止幻觉情况，把温度调到最低（极大减少幻觉情况产生）
        "max_tokens": 512,  
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    retry_count = 0
    max_retries = 5
    #这里我设定重连以及极端情况下的睡眠（防止访问api太频繁导致失败数据丢失，所以这里加上对应重试次数，这个阈值多次测试都是很保险的可以不动）
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
                "category_input": res.get('category', '')  # 原始分类
            }
            # 填充提取字段
            for k, v in FIELD_MAP.items():
                row[v] = res.get(k, '')
            writer.writerow(row)
            
    print(f"\n完成! 结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    #这里防止设备不同，加上一个限定（因为在我自己的设备上是MAC,这里公司设备使用win32是适配的）
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())