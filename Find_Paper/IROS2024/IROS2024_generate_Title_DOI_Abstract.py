"""
根据DOI提取IROS 2024论文摘要 (使用代理，带进度条和时间估计)
"""
import requests
import re
import csv
import time
from tqdm import tqdm

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
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    total_papers = len(rows) - 1
    print(f"共需处理 {total_papers} 篇论文")
    print("测试模式：只处理前20篇论文")

    # 跳过表头，处理每一行，只处理前20篇
    max_papers = total_papers
    processed_papers = 0
    total_time = 0

    # 使用tqdm创建进度条
    with tqdm(total=max_papers, desc="处理论文摘要", unit="篇") as pbar:
        for i in range(1, max_papers + 1):
            start_time = time.time()
            title = rows[i][0]
            doi = rows[i][1]

            # 确保行有3列
            while len(rows[i]) < 3:
                rows[i].append("")

            # 如果已经有摘要，跳过
            if rows[i][2]:
                print(f"  已有摘要，跳过: {title[:50]}...")
                pbar.update(1)
                continue

            # 提取摘要
            abstract = extract_abstract_from_doi(doi)

            if abstract:
                rows[i][2] = abstract
                print(f"  摘要提取成功 ({len(abstract)} 字符): {title[:50]}...")
            else:
                rows[i][2] = ""
                print(f"  未找到摘要: {title[:50]}...")

            # 每处理一篇就保存一次，避免数据丢失
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

            # 计算处理时间
            processing_time = time.time() - start_time
            total_time += processing_time
            processed_papers += 1

            # 估计剩余时间
            if processed_papers > 0:
                avg_time_per_paper = total_time / processed_papers
                remaining_papers = max_papers - i
                estimated_remaining_time = avg_time_per_paper * remaining_papers
                pbar.set_postfix_str(f"剩余时间: {estimated_remaining_time:.2f}秒")

            # 延迟避免请求过快
            time.sleep(2)
            pbar.update(1)

    print(f"\n测试完成！已处理前 {max_papers} 篇论文")
    print(f"结果已保存到: {output_file}")
    print("\n要运行完整版本，请修改脚本中的 max_papers 变量，将其设置为 total_papers")
    print("完整版本命令:")
    print("conda activate Test_lx && python /home/cuhk/Documents/Test_lx/Find_Paper/IROS2024/IROS2024_generate_Title_DOI_Abstract.py")

if __name__ == "__main__":
    scrape_abstracts()