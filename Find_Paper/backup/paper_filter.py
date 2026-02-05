# medical_paper_filter_batch_processor_FIXED_V5.py
import json
import os
import re
import math
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools


# ===========================
# 🧠 提示词（Prompt）配置区（请在此处修改判断逻辑）
# ===========================

JSON_OUTPUT_TEMPLATE = {
    "Title": "论文原标题占位符",
    "Title Translation": "标题的中文翻译",
    "MedicalDiagnosisPrognosisRelevance": "高/中/低",
    "Reason1": "简短理由（50字以内）：解释与医学诊断/预后相关的程度。",
    "FewZeroShotRelevance": "高/中/低",
    "Reason2": "简短理由（50字以内）：解释与少样本/零样本相关的程度。",
    "BodyPart": "肺部/脑/乳腺/眼底/皮肤/心脏/通用/其他",
    "Reason3": "简短理由（50字以内）：解释涉及的身体部位。",
    "Recommendation": "强烈推荐/一般推荐/不推荐"
}

SYSTEM_PROMPT = f'''你是一名医学人工智能方向的科研助理，当前任务是根据论文标题对其进行详细分析。
请严格分析以下三个核心方面：
1. **医学诊断/预后相关性**：是否与医疗/医学图像、诊断 (Diagnosis)、预后 (Prognosis) 相关。
2. **少/零样本相关性**：是否与少样本 (Few-shot)、零样本 (Zero-shot) 或迁移学习相关。
3. **涉及的身体部位**：明确提及或强烈暗示的操作对象。

请严格仅输出 JSON 格式的字符串，不要输出任何额外说明或 Markdown 格式。
你必须严格遵循以下 **JSON** 格式输出，Title 字段必须是输入的原标题：
{json.dumps(JSON_OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)}
'''


# ===========================
# 核心函数：DeepSeek 论文标题判断（线程安全）
# ===========================

def deepseek_judge_paper_json(title: str, api_key: str) -> Dict[str, Any]:
    user_prompt = f"论文标题：{title}"
    
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
        result['Original_Title'] = title
        return result

    except Exception as e:
        error_msg = f"API 或解析失败: {str(e)[:50]}"
        print(f"❌ 错误: {error_msg} (标题: {title[:30]}...)")
        return {
            "Original_Title": title,
            "Title": title,
            "Title Translation": "N/A",
            "MedicalDiagnosisPrognosisRelevance": "低",
            "Reason1": error_msg,
            "FewZeroShotRelevance": "低",
            "Reason2": "N/A",
            "BodyPart": "N/A",
            "Reason3": "N/A",
            "Recommendation": "不推荐"
        }


# ===========================
# 批量处理 Excel：自动输出路径 + 断点续传 + 并发
# ===========================

def batch_process_excel(input_path: str, api_keys: List[str]):
    # 自动构造输出路径：同目录，文件名加 _ds 后缀
    input_dir = os.path.dirname(input_path)
    input_basename = os.path.basename(input_path)
    name, ext = os.path.splitext(input_basename)
    output_path = os.path.join(input_dir, f"{name}_ds{ext}")

    ORIGINAL_COLS = ["Part", "Title", "Pages"]
    RESULT_COL_MAP = {
        "Title Translation": "Title Translation",
        "MedicalDiagnosisPrognosisRelevance": "MedicalDiagnosisPrognosisRelevance",
        "Reason1": "Reason1",
        "FewZeroShotRelevance": "FewZeroShotRelevance",
        "Reason2": "Reason2",
        "BodyPart": "BodyPart",
        "Reason3": "Reason3",
        "Recommendation": "Recommendation"
    }
    RESULT_COLS = list(RESULT_COL_MAP.values())

    # 确保输出目录存在（虽然与输入同目录，但保持健壮性）
    os.makedirs(input_dir, exist_ok=True)

    # 1. 读取原始数据
    df_original = pd.read_excel(input_path)
    if "Title" not in df_original.columns:
        raise ValueError("输入 Excel 文件缺少 'Title' 列")

    # 2. 尝试加载已有结果（用于续传）
    df_results = None
    if os.path.exists(output_path):
        print(f"✅ 检测到已有输出文件，尝试读取已处理结果：{output_path}")
        df_results = pd.read_excel(output_path)
        if "Title" not in df_results.columns:
            print("⚠️ 警告: 输出文件缺少 'Title' 列，将忽略已存在的结果。")
            df_results = None

    # 3. 初始化最终 DataFrame
    df_final = df_original.copy()
    for col in RESULT_COLS:
        if col not in df_final.columns:
            df_final[col] = None

    # 4. 同步已有结果
    synced_count = 0
    if df_results is not None:
        processed_data = df_results.set_index('Title')[RESULT_COLS]
        for idx, row in df_final.iterrows():
            title = str(row.get("Title", "")).strip()
            if not title or title.lower() in ("nan", "none"):
                continue
            if title in processed_data.index:
                rec = processed_data.loc[title, 'Recommendation']
                is_filled = (
                    isinstance(rec, str)
                    and rec.strip().lower() not in ["", "none", "nan", "n/a"]
                    and pd.notna(rec)
                )
                if is_filled:
                    for col in RESULT_COLS:
                        df_final.loc[idx, col] = processed_data.loc[title, col]
                    synced_count += 1
        print(f"   已同步 {synced_count} 条已处理结果到当前批次。")

    # 5. 收集待处理任务
    tasks_to_run = []
    total_rows = len(df_final)
    for idx, row in df_final.iterrows():
        title = str(row.get("Title", "")).strip()
        if not title or title.lower() in ("nan", "none"):
            continue

        recommendation_value = row.get("Recommendation")
        is_processed = False
        if isinstance(recommendation_value, str) and recommendation_value.strip().lower() not in ["", "none", "nan", "n/a"]:
            is_processed = True
        elif pd.notna(recommendation_value):
            is_processed = False

        if is_processed:
            print(f"⏩ 跳过已处理 [{idx+1}/{total_rows}]: {title[:50]}...")
            continue

        tasks_to_run.append({'index': idx, 'title': title})

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
            key = next(api_key_cycler)
            future = executor.submit(deepseek_judge_paper_json, title, key)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results_list.append((idx, result))
                print(f"✔ 完成 [{idx+1}/{total_rows}] | 结果: {result.get('Recommendation', 'N/A')} | 标题: {result.get('Title', 'N/A')[:80]}...")
            except Exception as e:
                print(f"❌ 线程执行错误 (Index: {idx}): {e}")

    # 7. 更新并保存
    for idx, result in results_list:
        for json_key, excel_col in RESULT_COL_MAP.items():
            df_final.at[idx, excel_col] = result.get(json_key, "N/A")

    df_final.to_excel(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"🎉 全部处理完成！共处理 {rows_to_process} 行新数据。")
    print(f"📝 最终输出文件：{output_path}")
    print(f"{'='*60}")


# ======================
# 主程序入口
# ======================
if __name__ == "__main__":
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

    INPUT_EXCEL = r"E:OneDrive - CUHK-ShenzhenOutside School2511MED_Interncodefind_paperoutputgoogle_scholar_output.xlsx"

    batch_process_excel(INPUT_EXCEL, API_KEYS)