"""
爬取DBLP网站2024年ICRA会议论文标题和DOI链接
"""
import requests
from bs4 import BeautifulSoup
import csv

def scrape_icra_titles_with_links():
    """爬取ICRA 2024论文标题和DOI链接"""
    url = "https://dblp.org/db/conf/icra/icra2024.html"

    # 设置请求头，模拟浏览器访问
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"正在访问: {url}")

    # 发送请求
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # 解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有论文条目 (<li class="entry inproceedings">)
    paper_entries = soup.find_all('li', class_='entry inproceedings')

    results = []

    for entry in paper_entries:
        # 提取标题
        title_elem = entry.find('span', class_='title')
        title = title_elem.get_text(strip=True) if title_elem else ""

        # 提取DOI链接 (在 <li class="drop-down"><div class="head"><a> 中)
        doi_link = ""
        dropdown = entry.find('li', class_='drop-down')
        if dropdown:
            head_div = dropdown.find('div', class_='head')
            if head_div:
                link_elem = head_div.find('a')
                if link_elem and link_elem.get('href'):
                    doi_link = link_elem.get('href')

        # 过滤掉会议标题等非论文内容
        if title and "IEEE International Conference" not in title:
            results.append({
                'title': title,
                'doi_link': doi_link
            })

    print(f"共找到 {len(results)} 篇论文")

    # 保存到CSV文件
    output_file = "/home/cuhk/Documents/Test_lx/Find_Paper/ICRA2024_Title_AbstractLink.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title', 'DOI_Link'])  # 写入表头
        for paper in results:
            writer.writerow([paper['title'], paper['doi_link']])

    print(f"已保存到: {output_file}")
    return results

if __name__ == "__main__":
    scrape_icra_titles_with_links()
