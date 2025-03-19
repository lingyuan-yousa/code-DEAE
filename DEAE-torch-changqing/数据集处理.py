import pandas as pd

# 读取CSV文件
df = pd.read_csv('changqing.csv', encoding='gbk')

# 将LITH列转换为类别类型并重新编号
df['LITH'] = df['LITH'].astype('category')
df['LITH'] = df['LITH'].cat.codes + 1

# 保存处理后的结果到原始CSV文件
df.to_csv('changqing.csv', index=False)
