import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import sys

# 定义目标数据集信息
target_species = 'Rapeseed'
target_phenotype = 'some_phenotype'  # 实际使用时需替换为具体表型名

target_stats = {
    'pheno_mean': 176.67,
    'pheno_std': 10.50,
    'pheno_skewness': -1.56,
    'pheno_kurtosis': 4.02,
    'pheno_min': None,  # 可根据实际数据补充
    'pheno_max': None,
    'pheno_median': None,
}

# 读取CSV文件
file_path = "/home/common/hwluo/project/gwas-llm-judge/experience/tmp/dataset_summary_masked_8a0d6dffd94440d5af851e56d5a764f6.csv"
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading file: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# 检查必要的列是否存在
required_columns = [
    'species', 'phenotype_name', 'is_pheno_binary',
    'pheno_mean', 'pheno_std', 'pheno_skewness', 'pheno_kurtosis',
    'pheno_min', 'pheno_max', 'pheno_median',
    'n_samples_total', 'n_markers'
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in CSV: {missing_columns}")

# 清洗数据：过滤掉二分类表型和目标数据集本身
filtered_df = df[(df['is_pheno_binary'] == False) & (df['species'] != target_species)]

# 提取关键统计字段
key_columns = [
    'species', 'phenotype_name',
    'pheno_mean', 'pheno_std', 'pheno_skewness', 'pheno_kurtosis',
    'pheno_min', 'pheno_max', 'pheno_median',
    'n_samples_total', 'n_markers'
]

# 构造特征矩阵
feature_cols = ['pheno_mean', 'pheno_std', 'pheno_skewness', 'pheno_kurtosis',
                'pheno_min', 'pheno_max', 'pheno_median', 'n_samples_total', 'n_markers']

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(filtered_df[feature_cols])

# 目标数据集标准化
target_features = np.array([[target_stats[col] for col in feature_cols]])
# 假设 target_stats 中的值已按相同顺序排列
if len(target_features[0]) != len(feature_cols):
    raise ValueError("Target stats do not match expected feature columns")

# 标准化目标特征
target_features_scaled = scaler.transform(target_features)

# 计算余弦相似度
similarities = cosine_similarity(target_features_scaled, features_scaled)[0]

# 获取排序后的索引
sorted_indices = np.argsort(similarities)[::-1]  # 降序排列

# 输出前五个最相似的数据集
results = []
for i in sorted_indices[:5]:
    row = filtered_df.iloc[i]
    result_str = f"{row['species']}/{row['phenotype_name']}"
    results.append(result_str)

print(results)