import numpy as np
from scipy.stats import skew, kurtosis
import os

def process_one_phenotype(dataset_path:str) -> dict:
    """
    处理单个表型，返回 summary 字典
    """
    geno_path = os.path.join(dataset_path, "genotype.npz")
    pheno_path = os.path.join(dataset_path, "phenotype.npz")

    genotype = np.load(geno_path)['arr_0']
    pheno_file = np.load(pheno_path)
    phenotype = pheno_file['arr_0']
    phe_name = pheno_file['arr_1']
    sp_name = pheno_file['arr_2']
    phe_data = phenotype[:, 0]

    # 去除缺失值
    mask = ~np.isnan(phe_data)
    phe_clean = phe_data[mask]
    geno_clean = genotype[mask] if mask.sum() > 0 else genotype

    summary = {
        # 基本信息
        # 'species_phenotype': f"{sp_name}/{phe_name}",
        'species': sp_name,
        # 'phenotype_name': phe_name,

        # 维度信息
        'n_samples_total': genotype.shape[0],
        'n_samples_valid': len(phe_clean),
        'n_markers': genotype.shape[1] if genotype.ndim > 1 else 1,
        'missing_rate': 1 - len(phe_clean) / genotype.shape[0],

        # 表型统计特征
        'pheno_mean': np.mean(phe_clean) if len(phe_clean) > 0 else np.nan,
        'pheno_std': np.std(phe_clean) if len(phe_clean) > 0 else np.nan,
        'pheno_min': np.min(phe_clean) if len(phe_clean) > 0 else np.nan,
        'pheno_max': np.max(phe_clean) if len(phe_clean) > 0 else np.nan,
        'pheno_median': np.median(phe_clean) if len(phe_clean) > 0 else np.nan,
        'pheno_skewness': skew(phe_clean) if len(phe_clean) > 3 else np.nan,
        'pheno_kurtosis': kurtosis(phe_clean) if len(phe_clean) > 3 else np.nan,

        # 基因型统计特征
        'geno_mean': np.mean(geno_clean) if geno_clean.size > 0 else np.nan,
        'geno_std': np.std(geno_clean) if geno_clean.size > 0 else np.nan,
        'geno_missing_rate': (
            np.isnan(geno_clean).sum() / geno_clean.size
            if geno_clean.size > 0 else np.nan
        ),
        'geno_maf': (
            np.mean(
                np.minimum(
                    np.mean(geno_clean, axis=0),
                    1 - np.mean(geno_clean, axis=0)
                )
            ) if geno_clean.ndim > 1 and geno_clean.size > 0 else np.nan
        ),

        # 类型信息
        'geno_dtype': str(genotype.dtype),
        'pheno_dtype': str(phe_data.dtype),
        'is_pheno_binary': len(np.unique(phe_clean)) == 2 if len(phe_clean) > 0 else False
    }

    return summary