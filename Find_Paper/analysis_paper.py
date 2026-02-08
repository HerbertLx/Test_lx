import json
import os
import random
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from tqdm import tqdm

# ===========================
# 🤖 提示词（Prompt）配置区
# ===========================

# 翻译模板
TRANSLATION_TEMPLATE = {
    "标题": "标题的中文翻译",
    "摘要": "摘要的中文翻译"
}

# 分析模板
ANALYSIS_TEMPLATE = {
    "title": "论文标题",
    "extracted_keywords": ["关键词1", "关键词2", "..."],
    "classification": {
        "platform": "选定的标签",
        "methodology": "选定的标签",
        "application": "选定的标签"
    },
    "summary": "用一句话精炼概括核心贡献"
}

# 翻译提示词
TRANSLATION_PROMPT = '''你是一名机器人领域的科研助理，当前任务是将论文标题和摘要翻译成中文。

输入格式：
论文标题：[英文标题]
摘要：[英文摘要]

请严格执行以下任务：
1. **翻译**：将英文标题和摘要翻译成中文

请严格仅输出 JSON 格式的字符串，不要输出任何额外说明或 Markdown 格式。
你必须严格遵循以下 JSON 格式输出：
{
  "标题": "标题的中文翻译",
  "摘要": "摘要的中文翻译"
}
'''

# 分析提示词
ANALYSIS_PROMPT = '''你是一个机器人领域资深专家，请阅读论文标题与摘要，执行以下任务：

输入格式：
论文标题：[英文标题]
摘要：[英文摘要]

Task 1: 关键词提取 自主提取 3-5 个反映论文核心技术、对象或场景的中文关键词。

Task 2: 标准化分类 基于提取的理解，从以下三个维度各选出一个最匹配的标签（若不匹配或未提及，统一归类为"其它"）：

硬件平台 (Platform): 机械臂、人形机器人、足式机器人、无人机、智能小车、水下/空间机器人、医疗/康复机器人、柔性/微纳机器人、其它。

技术方法 (Methodology): 强化学习、模仿学习、大模型/具身AI、SLAM/导航、计算机视觉/感知、路径规划、控制理论、人机交互、多机协同、其它。

应用场景 (Application): 工业制造、物流仓储、家庭服务、野外/搜救、自动驾驶、农业/环境、手术/诊疗、教育/娱乐、其它。

请严格仅输出 JSON 格式的字符串，不要输出任何额外说明或 Markdown 格式。
你必须严格遵循以下 JSON 格式输出：
{
  "title": "论文标题",
  "extracted_keywords": ["关键词1", "关键词2", "..."],
  "classification": {
    "platform": "选定的标签",
    "methodology": "选定的标签",
    "application": "选定的标签"
  },
  "summary": "用一句话精炼概括核心贡献"
}
'''

# ===========================
# 核心函数：DeepSeek 论文分析（线程安全）
# ===========================

def deepseek_translate_paper(title: str, abstract: str, api_key: str) -> Dict[str, Any]:
    """使用DeepSeek API翻译论文标题和摘要"""
    user_prompt = f"论文标题：{title}\n摘要：{abstract}"
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": TRANSLATION_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={'type': 'json_object'},
            stream=False,
            temperature=0.0
        )
        raw_output = response.choices[0].message.content.strip()
        result = json.loads(raw_output)
        return result

    except Exception as e:
        return {
            "标题": f"[翻译] {title}",
            "摘要": f"[翻译] {abstract[:100]}..."
        }


def deepseek_analyze_paper_json(title: str, abstract: str, api_key: str) -> Dict[str, Any]:
    """使用DeepSeek API分析论文"""
    user_prompt = f"论文标题：{title}\n摘要：{abstract}"
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={'type': 'json_object'},
            stream=False,
            temperature=0.0
        )
        raw_output = response.choices[0].message.content.strip()
        result = json.loads(raw_output)
        result['title'] = title
        return result

    except Exception as e:
        return {
            "title": title,
            "extracted_keywords": ["N/A"],
            "classification": {
                "platform": "其它",
                "methodology": "其它",
                "application": "其它"
            },
            "summary": "分析失败"
        }

# ===========================
# 批量处理 CSV：自动输出路径 + 断点续传 + 并发
# ===========================

def batch_process_csv(input_path: str, output_path: str, api_keys: List[str], test_mode: bool = False, test_size: int = 10):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取原始数据
    df_original = pd.read_csv(input_path)
    if "Title" not in df_original.columns or "Abstract" not in df_original.columns:
        raise ValueError("输入 CSV 文件缺少必要列：Title, Abstract")

    # 2. 尝试加载已有结果（用于续传）
    df_results = None
    if os.path.exists(output_path):
        print(f"✅ 检测到已有输出文件，尝试读取已处理结果：{output_path}")
        df_results = pd.read_csv(output_path)
        # 清理列名中的空格
        df_results.columns = df_results.columns.str.strip()
        if "title" not in df_results.columns:
            print("⚠️ 警告: 输出文件缺少 'title' 列，将忽略已存在的结果。")
            df_results = None

    # 3. 初始化最终 DataFrame
    df_final = df_original.copy()
    # 确保列名格式正确
    desired_columns = ["标题", "摘要", "关键词", "平台", "方法", "应用场景", "总结"]
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
                "摘要": row.get("摘要", ""),
                "关键词": row.get("关键词", "N/A"),
                "平台": row.get("平台", "其它"),
                "方法": row.get("方法", "其它"),
                "应用场景": row.get("应用场景", "其它"),
                "总结": row.get("总结", "N/A")
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
                df_final.loc[idx, "关键词"] = data.get("关键词", "N/A")
                df_final.loc[idx, "平台"] = data.get("平台", "其它")
                df_final.loc[idx, "方法"] = data.get("方法", "其它")
                df_final.loc[idx, "应用场景"] = data.get("应用场景", "其它")
                df_final.loc[idx, "总结"] = data.get("总结", "N/A")
                synced_count += 1
        print(f"   已同步 {synced_count} 条已处理结果到当前批次。")

    # 5. 收集待处理任务
    tasks_to_run = []
    total_rows = len(df_final)
    skipped_rows = 0
    need_process_count = 0
    
    print(f"\n🔍 扫描 {total_rows} 行数据，筛选需要处理的任务...")
    
    for idx, row in df_final.iterrows():
        # 检查Title和Abstract是否完整
        title = str(row.get("Title", "")).strip()
        abstract = str(row.get("Abstract", "")).strip()
        
        if not title or not abstract:
            skipped_rows += 1
            continue

        # 检查是否有任何缺少的内容
        need_process = False
        
        # 检查翻译
        if pd.isna(row.get("标题")) or str(row.get("标题")).strip() == "":
            need_process = True
        if pd.isna(row.get("摘要")) or str(row.get("摘要")).strip() == "":
            need_process = True
        
        # 检查分析
        if pd.isna(row.get("关键词")) or str(row.get("关键词")).strip() == "":
            need_process = True
        if pd.isna(row.get("平台")) or str(row.get("平台")).strip() == "":
            need_process = True
        if pd.isna(row.get("方法")) or str(row.get("方法")).strip() == "":
            need_process = True
        if pd.isna(row.get("应用场景")) or str(row.get("应用场景")).strip() == "":
            need_process = True
        if pd.isna(row.get("总结")) or str(row.get("总结")).strip() == "":
            need_process = True

        if need_process:
            tasks_to_run.append({'index': idx, 'title': title, 'abstract': abstract})
            need_process_count += 1

    print(f"   跳过 {skipped_rows} 行（Title或Abstract不完整）")
    print(f"   需要处理 {need_process_count} 行（缺少翻译或分析结果）")

    # 测试模式：按顺序取前N个任务
    if test_mode and tasks_to_run:
        print(f"\n🎯 测试模式：从 {len(tasks_to_run)} 个任务中按顺序取前 {test_size} 个进行测试")
        tasks_to_run = tasks_to_run[:min(test_size, len(tasks_to_run))]

    rows_to_process = len(tasks_to_run)
    print(f"\n--- 待处理总行数: {total_rows} | 实际处理任务数: {rows_to_process} ---")

    if rows_to_process == 0:
        print("🎉 所有数据均已处理完成，无需运行新任务。")
        return

    # 6. 并发处理
    api_key_cycler = itertools.cycle(api_keys)
    max_workers = min(len(api_keys), 10)  # 防止线程过多（可调）
    results_list = []
    success_count = 0
    failure_count = 0

    print(f"\n🚀 开始处理 {rows_to_process} 个任务...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task in tasks_to_run:
            idx = task['index']
            title = task['title']
            abstract = task['abstract']
            key = next(api_key_cycler)
            
            # 先翻译，再分析
            def process_task(idx, title, abstract, key):
                # 翻译
                try:
                    translation = deepseek_translate_paper(title, abstract, key)
                except Exception as e:
                    translation = {"标题": f"[翻译] {title}", "摘要": f"[翻译] {abstract[:100]}..."}
                # 分析
                try:
                    analysis = deepseek_analyze_paper_json(title, abstract, key)
                except Exception as e:
                    analysis = {
                        "title": title,
                        "extracted_keywords": ["N/A"],
                        "classification": {
                            "platform": "其它",
                            "methodology": "其它",
                            "application": "其它"
                        },
                        "summary": "分析失败"
                    }
                return idx, translation, analysis
            
            future = executor.submit(process_task, idx, title, abstract, key)
            futures[future] = idx

        # 使用tqdm显示进度
        with tqdm(total=rows_to_process, desc="处理进度", unit="篇", ncols=100) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    idx, translation, analysis = future.result()
                    results_list.append((idx, translation, analysis))
                    success_count += 1
                    # 显示当前处理的行号和平台
                    platform = analysis.get('classification', {}).get('platform', '其它')
                    pbar.update(1)
                    pbar.set_postfix({"行号": idx+1, "平台": platform, "剩余": rows_to_process - (success_count + failure_count)})
                except Exception as e:
                    failure_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"行号": idx+1, "状态": "失败", "剩余": rows_to_process - (success_count + failure_count)})

    # 显示处理统计
    print(f"\n📊 处理统计:")
    print(f"   总任务数: {rows_to_process}")
    print(f"   成功: {success_count}")
    print(f"   失败: {failure_count}")

    # 7. 更新并保存
    print(f"\n💾 更新并保存结果...")
    for idx, translation, analysis in results_list:
        # 更新翻译
        df_final.at[idx, "标题"] = translation.get("标题", "")
        df_final.at[idx, "摘要"] = translation.get("摘要", "")
        # 更新分析
        df_final.at[idx, "关键词"] = ", ".join(analysis.get("extracted_keywords", ["N/A"]))
        df_final.at[idx, "平台"] = analysis.get("classification", {}).get("platform", "其它")
        df_final.at[idx, "方法"] = analysis.get("classification", {}).get("methodology", "其它")
        df_final.at[idx, "应用场景"] = analysis.get("classification", {}).get("application", "其它")
        df_final.at[idx, "总结"] = analysis.get("summary", "N/A")

    # 8. 保存结果
    # 只保存需要的列
    output_columns = ["Title", "DOI", "Abstract", "标题", "摘要", "关键词", "平台", "方法", "应用场景", "总结"]
    df_output = df_final[[col for col in output_columns if col in df_final.columns]]
    
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 9. 检查空单元格
    print(f"\n🔍 检查空单元格...")
    
    # 读取保存后的文件进行检查
    df_check = pd.read_csv(output_path)
    df_check.columns = df_check.columns.str.strip()
    
    # 目标列
    target_columns = ["标题", "摘要", "关键词", "平台", "方法", "应用场景", "总结"]
    
    # 统计空单元格
    empty_cells = {}
    total_empty = 0
    
    for col in target_columns:
        if col in df_check.columns:
            # 检查空值
            empty_count = df_check[col].isna().sum()
            # 检查空字符串
            empty_string_count = df_check[col].astype(str).str.strip().eq('').sum()
            # 检查"N/A"值
            na_count = df_check[col].astype(str).str.strip().eq('N/A').sum()
            
            # 总空值数
            total_empty_in_col = empty_count + empty_string_count + na_count
            empty_cells[col] = total_empty_in_col
            total_empty += total_empty_in_col
    
    # 输出检查结果
    print(f"\n📊 空单元格检查结果:")
    print(f"   目标运行行数: {total_rows}")
    print(f"   实际处理行数: {rows_to_process}")
    print(f"   总空单元格数: {total_empty}")
    print(f"   各列空单元格数:")
    for col, count in empty_cells.items():
        print(f"      {col}: {count}")
    
    # 检查具体哪些行有空单元格
    print(f"\n🔍 检查具体空单元格位置...")
    empty_rows = []
    
    for idx, row in df_check.iterrows():
        row_empty = False
        empty_cols = []
        
        for col in target_columns:
            if col in df_check.columns:
                value = row.get(col, "")
                if pd.isna(value) or str(value).strip() == "" or str(value).strip() == "N/A":
                    row_empty = True
                    empty_cols.append(col)
        
        if row_empty:
            empty_rows.append((idx+1, empty_cols))  # 行号从1开始
    
    # 输出有空单元格的行
    if empty_rows:
        print(f"\n⚠️ 发现 {len(empty_rows)} 行存在空单元格:")
        # 只输出前10行，避免输出过多
        for i, (row_num, cols) in enumerate(empty_rows[:10]):
            print(f"      行 {row_num}: 空列 - {', '.join(cols)}")
        if len(empty_rows) > 10:
            print(f"      ... 还有 {len(empty_rows) - 10} 行未显示")
    else:
        print(f"\n✅ 所有单元格均已填充，无空单元格！")
    
    print(f"\n{'='*60}")
    print(f"🎉 全部处理完成！共处理 {rows_to_process} 行数据。")
    print(f"📝 最终输出文件：{output_path}")
    print(f"{'='*60}")

# ======================  
# 主程序入口
# ======================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用DeepSeek API分析论文")
    parser.add_argument("conference", type=str, help="会议名称，如ICRA2024")
    parser.add_argument("--test", action="store_true", help="测试模式，只处理少量数据")
    parser.add_argument("--test-size", type=int, default=10, help="测试模式处理的数据量")
    
    args = parser.parse_args()
    conference = args.conference
    test_mode = args.test
    test_size = args.test_size
    
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
    input_dir = f"/home/cuhk/Documents/Test_lx/Find_Paper/{conference}"
    input_csv = f"{input_dir}/{conference}_Title_DOI_Abstract.csv"
    output_csv = f"{input_dir}/{conference}_Title_DOI_Abstract_标题_摘要_解析.csv"

    print(f"会议: {conference}")
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    print(f"测试模式: {test_mode}")
    if test_mode:
        print(f"测试数据量: {test_size}")
    print()

    # 运行批量处理
    batch_process_csv(input_csv, output_csv, API_KEYS, test_mode, test_size)
