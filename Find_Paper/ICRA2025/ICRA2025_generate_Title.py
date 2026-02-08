"""
爬取DBLP网站2025年ICRA会议论文标题
"""
import requests
from bs4 import BeautifulSoup
import csv

# 设置VPN代理
proxies = {
    'http': 'http://127.0.0.1:7897',
    'https': 'http://127.0.0.1:7897'
}

def scrape_icra_titles():
    """爬取ICRA 2025论文标题"""
    url = "https://dblp.org/db/conf/icra/icra2025.html"

    # 设置请求头，模拟浏览器访问
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"正在访问: {url}")

    # 发送请求，使用代理
    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"请求失败: {e}")
        return []

    # 解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有标题
    titles = []
    title_elements = soup.find_all('span', class_='title')

    for elem in title_elements:
        title = elem.get_text(strip=True)
        if title and "IEEE International Conference" not in title:
            titles.append(title)

    print(f"共找到 {len(titles)} 篇论文")

    # 保存到CSV文件
    output_file = "/home/cuhk/Documents/Test_lx/Find_Paper/ICRA2025/ICRA2025_Title.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title'])
        for title in titles:
            writer.writerow([title])

    print(f"已保存到: {output_file}")
    return titles

if __name__ == "__main__":
    scrape_icra_titles()