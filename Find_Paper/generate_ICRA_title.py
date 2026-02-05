"""
爬取DBLP网站2024年ICRA会议论文标题
"""
import requests
from bs4 import BeautifulSoup
import csv

def scrape_icra_titles():
    """爬取ICRA 2024论文标题"""
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

    # 查找所有标题
    titles = []
    title_elements = soup.find_all('span', class_='title')

    for elem in title_elements:
        title = elem.get_text(strip=True)
        if title:
            titles.append(title)

    print(f"共找到 {len(titles)} 篇论文")

    # 保存到CSV文件
    output_file = "/home/cuhk/Documents/Test_lx/Find_Paper/ICRA_title.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title'])  # 写入表头
        for title in titles:
            writer.writerow([title])

    print(f"已保存到: {output_file}")
    return titles

if __name__ == "__main__":
    scrape_icra_titles()
