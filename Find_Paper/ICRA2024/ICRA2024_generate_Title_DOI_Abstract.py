"""
根据DOI提取ICRA 2024论文摘要 (使用代理)
"""
import requests
import re
import csv
import time
import os

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
    input_file = "/home/cuhk/Documents/Test_lx/Find_Paper/ICRA2024/ICRA2024_Title_DOI_Abstract.csv"
    output_file = "/home/cuhk/Documents/Test_lx/Find_Paper/ICRA2024/ICRA2024_Title_DOI_Abstract.csv"

    # 读取CSV文件
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    print(f"共需处理 {len(rows) - 1} 篇论文")

    # 跳过表头，处理每一行
    for i in range(1, len(rows)):
        title = rows[i][0]
        doi = rows[i][1]

        print(f"[{i}/{len(rows)-1}] 正在处理: {title[:50]}...")

        # 确保行有3列
        while len(rows[i]) < 3:
            rows[i].append("")

        # 如果已经有摘要，跳过
        if rows[i][2]:
            print("  已有摘要，跳过")
            continue

        # 提取摘要
        abstract = extract_abstract_from_doi(doi)

        if abstract:
            rows[i][2] = abstract
            print(f"  摘要提取成功 ({len(abstract)} 字符)")
        else:
            rows[i][2] = ""
            print("  未找到摘要")

        # 每处理一篇就保存一次，避免数据丢失
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        # 延迟避免请求过快
        time.sleep(2)

    print(f"\n完成！结果已保存到: {output_file}")

if __name__ == "__main__":
    scrape_abstracts()
