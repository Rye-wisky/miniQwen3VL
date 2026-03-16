from datasets import load_dataset

# 加载本地 parquet 文件
dataset = load_dataset('parquet', data_files='pretrain_i2t.parquet', split='train')

# 查看第一条数据
print(dataset[0])

# 查看数据集结构（特征名和类型）
print(dataset.features)