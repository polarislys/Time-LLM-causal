import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('./dataset/dataset/ETT-small/ETTh1.csv')

# 检查数据分布
print("数据集大小:", len(df))
print("\n数据统计:")
print(df.describe())

# 检查是否有异常值
print("\n是否有 NaN:", df.isnull().sum().sum())
print("是否有 Inf:", np.isinf(df.select_dtypes(include=[np.number])).sum().sum())

# 查看数据分布
import matplotlib.pyplot as plt
df.iloc[:, 1:].plot(figsize=(15, 8))
plt.title('ETTh1 Data Distribution')
plt.savefig('data_distribution.png')
print("\n数据分布图已保存到 data_distribution.png")