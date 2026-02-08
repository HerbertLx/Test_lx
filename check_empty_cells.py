import csv
from collections import defaultdict

file_path = '/home/cuhk/Documents/Test_lx/Find_Paper/ICRA2024/ICRA2024_Title_DOI_Abstract_标题_摘要.csv'

# 使用csv模块读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

# 获取列名
headers = rows[0]
print('列名:', headers)
print()

# 统计每列的空白单元格数量
empty_counts = defaultdict(int)
total_rows = len(rows) - 1

print('每列的空白单元格数量:')
for i, row in enumerate(rows[1:], start=1):
    for j, cell in enumerate(row):
        if not cell or cell.strip() in ['', ' ', 'N/A', 'nan', 'None']:
            col_name = headers[j] if j < len(headers) else f'Column_{j}'
            empty_counts[col_name] += 1

for col, count in empty_counts.items():
    print(f'{col}: {count} ({count/total_rows*100:.2f}%)')

print(f'\n总行数: {total_rows}')
print('\n检查是否有列数不一致的行:')
inconsistent_rows = []
for i, row in enumerate(rows[1:], start=1):
    if len(row) != len(headers):
        inconsistent_rows.append(i)
        print(f'第{i}行: 列数={len(row)}, 预期列数={len(headers)}')

print(f'\n列数不一致的行数: {len(inconsistent_rows)}')

# 检查是否有完全空白的行
print('\n检查是否有完全空白的行:')
blank_rows = []
for i, row in enumerate(rows[1:], start=1):
    if all(not cell or cell.strip() == '' for cell in row):
        blank_rows.append(i)

if blank_rows:
    print(f'发现 {len(blank_rows)} 个完全空白的行: {blank_rows}')
else:
    print('没有发现完全空白的行')

# 预览前几行数据
print('\n前5行数据预览:')
for i, row in enumerate(rows[:5]):
    print(f'第{i}行: {row}')