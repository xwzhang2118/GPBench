import csv
import os
import uuid
from typing import List, Optional, Tuple


def get_matched_experience(
    target_species_phenotypes: Optional[List[str]],
    source_csv: Optional[str] = None,
    output_suffix: str = "_matched",
) -> Tuple[str, str]:
    """
    根据给定的 species/phenotype 列表，从 experience_origin.csv 中筛选出匹配的行，
    生成新的 CSV 文件，并返回新文件的绝对路径与前 10 行预览（包含表头，字符串形式）。

    参数
    ----
    target_species_phenotypes : List[str] | None
        需要保留的 species/phenotype 组合，格式如 "Cattle/fpro"。
        如果为 None，则不过滤，返回源 CSV 中的全部记录。
    source_csv : str, optional
        源 CSV 文件路径，默认指向当前脚本同目录下的 experience_origin.csv（相对路径）。
    output_suffix : str, optional
        生成的新文件名后缀，默认 "_matched"。

    返回
    ----
    tuple[str, str]
        新生成的 CSV 文件的绝对路径，以及匹配数据的前 10 行（含表头）序列化字符串。
    """
    base_dir = os.path.dirname(__file__)

    if source_csv is None:
        source_csv = os.path.join(base_dir, "experience_origin.csv")
    elif not os.path.isabs(source_csv):
        source_csv = os.path.join(base_dir, source_csv)

    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    # 使用小写的 (species, phenotype) 组合来做匹配，
    # 以实现大小写不敏感的匹配逻辑。
    match_set = set()
    if target_species_phenotypes is not None:
        for item in target_species_phenotypes:
            if "/" not in item:
                raise ValueError(f"Invalid format (expected species/phenotype): {item}")
            species, phenotype = item.split("/", 1)
            species = species.strip()
            phenotype = phenotype.strip()
            if not species or not phenotype:
                raise ValueError(f"Invalid species or phenotype in: {item}")
            # 统一转为小写存入集合
            match_set.add((species.lower(), phenotype.lower()))

    kept_rows: List[List[str]] = []
    header: Optional[List[str]] = None

    with open(source_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue

            if len(row) < 3:
                continue

            # 如果 target_species_phenotypes 为 None，则不过滤，保留所有记录
            if target_species_phenotypes is None:
                kept_rows.append(row)
            else:
                # 使用小写形式进行匹配，实现大小写不敏感
                species_val = row[1].strip().lower()
                pheno_val = row[2].strip().lower()
                if (species_val, pheno_val) in match_set:
                    kept_rows.append(row)

    preview_rows: List[List[str]] = []
    if header is not None:
        preview_rows.append(header)
    data_limit = max(0, 10 - len(preview_rows))
    if data_limit > 0:
        preview_rows.extend(kept_rows[:data_limit])
    preview_str = "\n".join([",".join(row) for row in preview_rows])

    tmp_dir = os.path.join(base_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    _, src_name = os.path.split(source_csv)
    name, ext = os.path.splitext(src_name)
    uid = uuid.uuid4().hex
    output_name = f"{name}{output_suffix}_{uid}{ext}"
    output_path = os.path.join(tmp_dir, output_name)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(kept_rows)

    return os.path.abspath(output_path), preview_str


if __name__ == "__main__":
    sample_targets = ["Rice/GYP_BLUP", "Mouse/weight", "Chickpea/Yield"]
    new_path, preview = get_matched_experience(sample_targets)
    print(f"Matched experience written to: {new_path}")
    print("Preview (first 10 rows):")
    print(preview)

