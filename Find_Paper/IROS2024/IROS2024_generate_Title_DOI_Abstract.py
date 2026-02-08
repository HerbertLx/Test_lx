"""
根据DOI提取IROS 2024论文摘要 (使用代理，带进度条和时间估计)
"""
import requests
import re
import csv
import time
import os
from tqdm import tqdm
import datetime

# 代理设置
PROXY = "http://127.0.0.1:7897"
PROXIES = {
    'http': PROXY,
    'https': PROXY,
}

def extract_abstract_from_doi(doi_url):
    """从DOI页面提取摘要"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    try:
        response = requests.get(doi_url, headers=headers, proxies=PROXIES, timeout=30)
        response.raise_for_status()
        html = response.text

        # 查找摘要 - 从 "abstract":" 开始
        abstract_pattern = r'"abstract"\s*:\s*"((?:[^"\\]|\\.)+)"'
        matches = re.findall(abstract_pattern, html)

        if matches:
            # 获取第一个非"true"的匹配
            for match in matches:
                if match != 'true' and len(match) > 10:
                    # 解码转义字符
                    abstract = match.encode().decode('unicode-escape')
                    return abstract

        return ""

    except Exception as e:
        print(f"  错误: {e}")
        return ""

def scrape_abstracts():
    """爬取所有论文的摘要"""
    input_file = "/home/cuhk/Documents/Test_lx/Find_Paper/IROS2024/IROS2024_Title_DOI.csv"
    output_file = "/home/cuhk/Documents/Test_lx/Find_Paper/IROS2024/IROS2024_Title_DOI_Abstract.csv"

    # 读取CSV文件
    rows = []
    
    # 优先从已有的摘要文件读取数据（如果存在）
    if os.path.exists(output_file):
        print(f"从已有文件读取数据: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        print(f"共 {len(rows) - 1} 篇论文，将只处理空白摘要")
    else:
        # 如果摘要文件不存在，从原始DOI文件读取
        print(f"从原始文件读取数据: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        print(f"共需处理 {len(rows) - 1} 篇论文")

    # 统计初始空摘要数量
    initial_empty_count = 0
    for i in range(1, len(rows)):
        while len(rows[i]) < 3:
            rows[i].append("")
        abstract = rows[i][2]
        if not (abstract and abstract.strip()):
            initial_empty_count += 1
    
    print(f"初始空白摘要数量: {initial_empty_count}")

    # 跳过表头，处理每一行
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    start_time = time.time()
    
    # 计算需要处理的任务数
    total_tasks = initial_empty_count
    
    # 使用tqdm添加进度条
    with tqdm(total=len(rows)-1, desc="处理进度", unit="篇") as pbar:
        for i in range(1, len(rows)):
            # 确保行有足够的列
            while len(rows[i]) < 3:
                rows[i].append("")
            
            title = rows[i][0]
            doi = rows[i][1]
            abstract = rows[i][2]

            # 如果已经有摘要，跳过
            if abstract and abstract.strip():
                skipped_count += 1
                pbar.update(1)
                continue

            # 提取摘要
            new_abstract = extract_abstract_from_doi(doi)

            if new_abstract:
                rows[i][2] = new_abstract
                processed_count += 1
            else:
                rows[i][2] = ""
                failed_count += 1

            # 每处理一篇就保存一次，避免数据丢失
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

            # 更新进度条
            pbar.update(1)
            
            # 估计剩余时间
            elapsed_time = time.time() - start_time
            processed_tasks = processed_count + failed_count
            if processed_tasks > 0:
                avg_time_per_task = elapsed_time / processed_tasks
                remaining_tasks = total_tasks - processed_tasks
                remaining_time = avg_time_per_task * remaining_tasks
                
                # 更新进度条描述
                eta_str = str(datetime.timedelta(seconds=int(remaining_time)))
                pbar.set_postfix({"已处理": processed_count, "失败": failed_count, "ETA": eta_str})

            # 延迟避免请求过快
            time.sleep(2)

    # 统计最终空摘要数量
    final_empty_count = 0
    for i in range(1, len(rows)):
        while len(rows[i]) < 3:
            rows[i].append("")
        abstract = rows[i][2]
        if not (abstract and abstract.strip()):
            final_empty_count += 1

    print(f"\n{'='*60}")
    print("处理完成！")
    print(f"{'='*60}")
    print(f"总论文数: {len(rows) - 1}")
    print(f"初始空白摘要: {initial_empty_count}")
    print(f"最终空白摘要: {final_empty_count}")
    print(f"成功处理: {processed_count}")
    print(f"处理失败: {failed_count}")
    print(f"跳过处理: {skipped_count}")
    print(f"{'='*60}")
    print(f"结果已保存到: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    scrape_abstracts()