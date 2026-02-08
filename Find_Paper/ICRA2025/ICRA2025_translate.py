"""
翻译ICRA2025论文标题和摘要
"""
import json
import os
import time
import itertools
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ===========================
# 🤖 提示词（Prompt）配置区
# ===========================

JSON_OUTPUT_TEMPLATE = {
    "Title": "论文原标题占位符",
    "标题": "标题的中文翻译",
    "DOI": "DOI占位符",
    "Abstract": "摘要原文本占位符",
    "摘要": "摘要的中文翻译"
}

SYSTEM_PROMPT = f'''你是一名专业的翻译助手，擅长将英文科技论文翻译成中文。
请严格执行以下任务：
1. **翻译**：将英文标题和摘要翻译成中文
2. **保持专业性**：翻译要准确反映原文的专业术语和内容
3. **流畅自然**：翻译后的中文要通顺易懂

请严格仅输出 JSON 格式的字符串，不要输出任何额外说明或 Markdown 格式。
你必须严格遵循以下 **JSON** 格式输出：
{json.dumps(JSON_OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)}
'''

# ===========================
# 核心函数：DeepSeek 翻译（线程安全）
# ===========================

def deepseek_translate_paper_json(title: str, doi: str, abstract: str, api_key: str) -> Dict[str, Any]:
    # 处理空摘要的情况
    if not abstract or abstract.strip() in ["", "N/A", "nan", "None"]:
        return {
            "Title": title,
            "标题": "",
            "DOI": doi,
            "Abstract": abstract,
            "摘要": ""
        }
    
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
            "标题": "",
            "DOI": doi,
            "Abstract": abstract,
            "摘要": ""
        }

# ===========================
# 批量处理 CSV：自动输出路径 + 断点续传
# ===========================

def batch_process_csv(input_path: str, output_path: str, api_keys: List[str]):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    RESULT_COL_MAP = {
        "标题": "标题",
        "摘要": "摘要"
    }
    RESULT_COLS = list(RESULT_COL_MAP.values())

    # 1. 读取原始数据
    # 使用names参数来指定列名，忽略第一行的列名
    df_original = pd.read_csv(input_path, names=['Title', 'DOI', 'Abstract'], skiprows=1)
    
    # 处理可能的空值
    df_original = df_original.fillna('')

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
    desired_columns = ["Title", "标题", "DOI", "Abstract", "摘要"]
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
                "标题": row.get("标题", ""),
                "摘要": row.get("摘要", "")
            }
        
        # 同步数据
        for idx, row in df_final.iterrows():
            title = str(row.get("Title", "")).strip()
            if not title or title.lower() in ("nan", "none"):
                continue
            if title in processed_data:
                data = processed_data[title]
                df_final.loc[idx, "标题"] = data.get("标题", "")
                df_final.loc[idx, "摘要"] = data.get("摘要", "")
                synced_count += 1
        print(f"   已同步 {synced_count} 条已处理结果到当前批次。")

    # 5. 收集待处理任务
    tasks_to_run = []
    total_rows = len(df_final)
    # 测试模式：只处理前10条
    max_tests = 10
    test_count = 0
    
    for idx, row in df_final.iterrows():
        title = str(row.get("Title", "")).strip()
        doi = str(row.get("DOI", "")).strip()
        abstract = str(row.get("Abstract", "")).strip()
        
        if not title or title.lower() in ("nan", "none"):
            continue

        # 检查是否已处理
        title_translated = row.get("标题")
        abstract_translated = row.get("摘要")
        is_processed = False
        if isinstance(title_translated, str) and title_translated.strip():
            is_processed = True

        if is_processed:
            print(f"⏩ 跳过已处理 [{idx+1}/{total_rows}]: {title[:50]}...")
            continue

        # 测试模式：只添加前10个未处理的
        if test_count < max_tests:
            tasks_to_run.append({'index': idx, 'title': title, 'doi': doi, 'abstract': abstract})
            test_count += 1
        else:
            break

    rows_to_process = len(tasks_to_run)
    print(f"\n--- 待处理总行数: {total_rows} | 新任务数: {rows_to_process} (测试模式) ---")

    if rows_to_process == 0:
        print("🎉 所有测试数据均已处理完成，无需运行新任务。")
        return

    # 6. 处理任务（带进度条和时间估计）
    api_key_cycler = itertools.cycle(api_keys)
    results_list = []
    total_time = 0

    # 使用tqdm创建进度条
    with tqdm(total=rows_to_process, desc="翻译论文", unit="篇") as pbar:
        for i, task in enumerate(tasks_to_run):
            start_time = time.time()
            idx = task['index']
            title = task['title']
            doi = task['doi']
            abstract = task['abstract']
            
            # 获取API密钥
            api_key = next(api_key_cycler)
            
            # 翻译论文
            result = deepseek_translate_paper_json(title, doi, abstract, api_key)
            results_list.append((idx, result))
            
            # 计算处理时间
            processing_time = time.time() - start_time
            total_time += processing_time
            
            # 估计剩余时间
            avg_time_per_paper = total_time / (i + 1)
            remaining_papers = rows_to_process - (i + 1)
            estimated_remaining_time = avg_time_per_paper * remaining_papers
            
            # 更新进度条
            pbar.set_postfix_str(f"剩余时间: {estimated_remaining_time:.2f}秒")
            pbar.update(1)
            
            # 显示处理结果
            print(f"✔ 完成 [{idx+1}/{total_rows}] | 标题: {result.get('Title', 'N/A')[:80]}...")

    # 7. 更新并保存
    for idx, result in results_list:
        for json_key, csv_col in RESULT_COL_MAP.items():
            df_final.at[idx, csv_col] = result.get(json_key, "")

    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"🎉 测试完成！共处理 {rows_to_process} 条数据。")
    print(f"📝 最终输出文件：{output_path}")
    print(f"{'='*60}")
    print("\n要运行完整版本，请修改脚本中的测试模式限制")

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
    INPUT_CSV = r"/home/cuhk/Documents/Test_lx/Find_Paper/ICRA2025/ICRA2025_Title_DOI_Abstract.csv"
    OUTPUT_CSV = r"/home/cuhk/Documents/Test_lx/Find_Paper/ICRA2025/ICRA2025_Title_DOI_Abstract_标题_摘要.csv"

    batch_process_csv(INPUT_CSV, OUTPUT_CSV, API_KEYS)