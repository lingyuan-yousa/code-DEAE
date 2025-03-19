import pandas as pd
from sklearn.model_selection import train_test_split
# 读取CSV文件
# df = pd.read_csv('changqing.csv', encoding='utf-8')

# 统计"Well_Name"列的唯一值数量
# unique_well_names = df['Well_Name'].unique()
#
# # 打印结果
# print(f"Well_Name列中有{unique_well_names}个不同的值。")
#
#
# print("Well_Name列中的不同值:")
# for well_name in unique_well_names:
#     print(well_name)

# num_rows, num_cols = df.shape
# print(num_rows)

# lith_value_counts = df['LITH'].value_counts()

# 打印结果
# print("LITH列中不同值的个数:")
# print(lith_value_counts)


df = pd.read_csv('changqing_中文.csv', encoding='gbk')
# # 根据"LITH"列的值进行分组，并计算每个分组的大小
# lith_group_sizes = df.groupby('LITH').size()
#
# # 打印结果
# print("每个值对应的数据条数:")
# print(lith_group_sizes)

# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
#
# # 打印划分结果的信息
# print(f"训练集样本数量: {len(df_train)}, 测试集样本数量: {len(df_test)}")

# 将LITH列转换为类别类型并获取对应的数字编码
df['LITH_CODE'] = df['LITH'].astype('category').cat.codes

# 获取每个类别对应的原始值和数字编码
category_mapping = df[['LITH', 'LITH_CODE']].drop_duplicates().sort_values('LITH_CODE').set_index('LITH_CODE')['LITH'].to_dict()

# 打印对应关系
print("数字编码和原始值的对应关系:")
for code, lith in category_mapping.items():
    print(f"{code}: {lith}")