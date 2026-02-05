import json
import os
import re
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

# ===========================
# 🤖 提示词（Prompt）配置区
# ===========================

JSON_OUTPUT_TEMPLATE = {
    "Title": "论文原标题占位符",
    "标题": "标题的中文翻译",
    "DOI": "DOI占位符",
    "Abstract": "摘要原文本占位符",
    "摘要": "摘要的中文翻译",
    "关键词": "提取的关键词，用分号分隔",
    "设备": "设备分类结果，如：机械臂/无人机/机器狗/人形机器人/无人小车/其他"
}

SYSTEM_PROMPT = f'''你是一名机器人领域的科研助理，当前任务是根据论文标题和摘要进行分析。
请严格执行以下任务：
1. **翻译**：将英文标题和摘要翻译成中文
2. **关键词提取**：从标题和摘要中提取5-8个核心关键词
3. **设备分类**：根据论文内容对涉及的设备进行分类，分类选项包括：
   - 机械臂
   - 无人机
   - 机器狗
   - 人形机器人
   - 无人小车
   - 其他

请严格仅输出 JSON 格式的字符串，不要输出任何额外说明或 Markdown 格式。
你必须严格遵循以下 **JSON** 格式输出：
{json.dumps(JSON_OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)}
'''

# ===========================
# 核心函数：DeepSeek 论文分析（线程安全）
# ===========================

def deepseek_analyze_paper_json(title: str, doi: str, abstract: str, api_key: str) -> Dict[str, Any]:
    user_prompt = f"论文标题：{title}\nDOI：{doi}\n摘要：{abstract}"
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={'type': 'json_object'},
            stream=False,
            temperature=0.0
        )
        raw_output = response.choices[0].message.content.strip()
        result = json.loads(raw_output)
        result['Title'] = title
        result['DOI'] = doi
        result['Abstract'] = abstract
        return result

    except Exception as e:
        error_msg = f"API 或解析失败: {str(e)[:50]}"
        print(f"❌ 错误: {error_msg} (标题: {title[:30]}...)")
        return {
            "Title": title,
            "标题": "N/A",
            "DOI": doi,
            "Abstract": abstract,
            "摘要": "N/A",
            "关键词": "N/A",
            "设备": "其他"
        }

# ===========================
# 批量处理 CSV：自动输出路径 + 断点续传 + 并发
# ===========================

def batch_process_csv(input_path: str, output_path: str, api_keys: List[str]):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    RESULT_COL_MAP = {
        "标题": "标题",
        "摘要": "摘要",
        "关键词": "关键词",
        "设备": "设备"
    }
    RESULT_COLS = list(RESULT_COL_MAP.values())

    # 1. 读取原始数据
    df_original = pd.read_csv(input_path)
    if "Title" not in df_original.columns or "DOI" not in df_original.columns or "Abstract" not in df_original.columns:
        raise ValueError("输入 CSV 文件缺少必要列：Title, DOI, Abstract")

    # 2. 尝试加载已有结果（用于续传）
    df_results = None
    if os.path.exists(output_path):
        print(f"✅ 检测到已有输出文件，尝试读取已处理结果：{output_path}")
        df_results = pd.read_csv(output_path)
        # 清理列名中的空格
        df_results.columns = df_results.columns.str.strip()
        if "Title" not in df_results.columns:
            print("⚠️ 警告: 输出文件缺少 'Title' 列，将忽略已存在的结果。")
            df_results = None

    # 3. 初始化最终 DataFrame
    df_final = df_original.copy()
    # 确保列名格式正确
    desired_columns = ["Title", "标题", "DOI", "Abstract", "摘要", "关键词", "设备"]
    for col in desired_columns:
        if col not in df_final.columns:
            df_final[col] = None

    # 4. 同步已有结果
    synced_count = 0
    if df_results is not None:
        # 构建处理数据的映射
        processed_data = {}
        for idx, row in df_results.iterrows():
            title = str(row.get("Title", "")).strip()
            if not title:
                continue
            processed_data[title] = {
                "标题": row.get("标题", "N/A"),
                "摘要": row.get("摘要", "N/A"),
                "关键词": row.get("关键词", "N/A"),
                "设备": row.get("设备", "其他")
            }
        
        # 同步数据
        for idx, row in df_final.iterrows():
            title = str(row.get("Title", "")).strip()
            if not title or title.lower() in ("nan", "none"):
                continue
            if title in processed_data:
                data = processed_data[title]
                device_value = data.get("设备", "")
                is_filled = (
                    isinstance(device_value, str)
                    and device_value.strip().lower() not in ["", "none", "nan", "n/a"]
                    and pd.notna(device_value)
                )
                if is_filled:
                    df_final.loc[idx, "标题"] = data.get("标题", "N/A")
                    df_final.loc[idx, "摘要"] = data.get("摘要", "N/A")
                    df_final.loc[idx, "关键词"] = data.get("关键词", "N/A")
                    df_final.loc[idx, "设备"] = data.get("设备", "其他")
                    synced_count += 1
        print(f"   已同步 {synced_count} 条已处理结果到当前批次。")

    # 5. 收集待处理任务
    tasks_to_run = []
    total_rows = len(df_final)
    for idx, row in df_final.iterrows():
        title = str(row.get("Title", "")).strip()
        doi = str(row.get("DOI", "")).strip()
        abstract = str(row.get("Abstract", "")).strip()
        
        if not title or title.lower() in ("nan", "none"):
            continue

        device_value = row.get("设备")
        is_processed = False
        if isinstance(device_value, str) and device_value.strip().lower() not in ["", "none", "nan", "n/a"]:
            is_processed = True
        elif pd.notna(device_value):
            is_processed = False

        if is_processed:
            print(f"⏩ 跳过已处理 [{idx+1}/{total_rows}]: {title[:50]}...")
            continue

        tasks_to_run.append({'index': idx, 'title': title, 'doi': doi, 'abstract': abstract})

    rows_to_process = len(tasks_to_run)
    print(f"\n--- 待处理总行数: {total_rows} | 新任务数: {rows_to_process} ---")

    if rows_to_process == 0:
        print("🎉 所有数据均已处理完成，无需运行新任务。")
        return

    # 6. 并发处理
    api_key_cycler = itertools.cycle(api_keys)
    max_workers = min(len(api_keys), 10)  # 防止线程过多（可调）
    results_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task in tasks_to_run:
            idx = task['index']
            title = task['title']
            doi = task['doi']
            abstract = task['abstract']
            key = next(api_key_cycler)
            future = executor.submit(deepseek_analyze_paper_json, title, doi, abstract, key)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results_list.append((idx, result))
                print(f"✔ 完成 [{idx+1}/{total_rows}] | 设备: {result.get('设备', 'N/A')} | 标题: {result.get('Title', 'N/A')[:80]}...")
            except Exception as e:
                print(f"❌ 线程执行错误 (Index: {idx}): {e}")

    # 7. 更新并保存
    for idx, result in results_list:
        for json_key, csv_col in RESULT_COL_MAP.items():
            df_final.at[idx, csv_col] = result.get(json_key, "N/A")

    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"🎉 全部处理完成！共处理 {rows_to_process} 行新数据。")
    print(f"📝 最终输出文件：{output_path}")
    print(f"{'='*60}")

# ======================
# 主程序入口
# ======================
if __name__ == "__main__":
    # API密钥列表
    API_KEYS = [
        "sk-8c38624bafb9477fb237ab2e58948c1b",
        "sk-4d5cc37a8b17417c87ed33b94d7b06a7",
        "sk-ab036985ba2c452f8262f4922e8ab50c",
        "sk-989bea94ce2b47e59714145005afc87e",
        "sk-29bc65dfb2de4ed3a983741f815ad2d7",
        "sk-064831a959c84353b6c4979fa90d16f3",
        "sk-82923973f2ff4ef895824595474f0df6",
        "sk-f0556cdf91f74729b7615d46fad4091c",
        "sk-6019eb2fda48482aa2218d3283f8332d",
        "sk-c19ad12b9f4b4b078270651858bb7f68",
    ]

    # 输入输出路径
    INPUT_CSV = r"/home/cuhk/Documents/Test_lx/Find_Paper/ICRA2024_Title_DOI_Abstract.csv"
    OUTPUT_CSV = r"/home/cuhk/Documents/Test_lx/Find_Paper/ICRA2024_Title_标题_DOI_Abstract_摘要_关键词.csv"

    batch_process_csv(INPUT_CSV, OUTPUT_CSV, API_KEYS)
