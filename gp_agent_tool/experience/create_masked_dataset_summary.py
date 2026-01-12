import csv
import os
import uuid
from typing import List, Optional


def create_masked_dataset_summary(
    excluded_species_phenotypes: List[str],
    source_csv: str = "dataset_summary.csv",
    output_suffix: str = "_masked",
) -> tuple[str, str]:
    """
    根据给定的 species_phenotype 列表，从 dataset_summary.csv 中过滤掉这些行，
    生成一个新的 CSV 文件，并返回新文件的绝对路径与前 10 行预览（包含表头）。

    参数
    ----
    excluded_species_phenotypes : List[str]
        需要被过滤掉的 species_phenotype 值列表（与源 CSV 中的第一列一致）。
    source_csv : str, optional
        源 CSV 文件路径，默认指向当前脚本同目录下的 dataset_summary.csv（相对路径）。
    output_suffix : str, optional
        生成的新文件名后缀，默认 "_masked"。

    返回
    ----
    tuple[str, str]
        新生成的 CSV 文件的绝对路径，以及 masked 数据的前 10 行（含表头）序列化字符串。
    """
    # 基于当前脚本位置来构造相对路径，避免依赖运行时工作目录
    base_dir = os.path.dirname(__file__)

    # 如果用户传入的是相对路径，则基于脚本目录解析（默认 dataset_summary.csv）
    if not os.path.isabs(source_csv):
        source_csv = os.path.join(base_dir, source_csv)

    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    # 读取源文件并过滤
    # 为了在匹配时实现“大小写不敏感”，预先构造一个全部转为小写的排除集合
    excluded_lower = {item.lower() for item in excluded_species_phenotypes}
    kept_rows: List[List[str]] = []
    header: Optional[List[str]] = None

    with open(source_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue

            # row[0] 应为 species_phenotype
            species_pheno = row[0]
            # 使用小写形式进行匹配，实现大小写不敏感
            if species_pheno.lower() in excluded_lower:
                continue
            kept_rows.append(row)

    # 预览前 10 行（含表头），序列化为字符串（逗号分隔，每行以 \n 拼接）
    preview_rows: List[List[str]] = []
    if header is not None:
        preview_rows.append(header)
    data_limit = max(0, 10 - len(preview_rows))
    if data_limit > 0:
        preview_rows.extend(kept_rows[:data_limit])
    preview_str = "\n".join([",".join(row) for row in preview_rows])

    # 构造输出文件路径：写到 experience/tmp 目录（相对脚本目录）
    tmp_dir = os.path.join(base_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    _, src_name = os.path.split(source_csv)
    name, ext = os.path.splitext(src_name)
    uid = uuid.uuid4().hex
    output_name = f"{name}{output_suffix}_{uid}{ext}"
    output_path = os.path.join(tmp_dir, output_name)

    # 写出新的 CSV 文件
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(kept_rows)

    return os.path.abspath(output_path), preview_str


if __name__ == "__main__":
    # 示例：排除若干 species_phenotype
    example_excluded = ["Cattle/mkg", "Chicken/EW28"]
    new_path, preview = create_masked_dataset_summary(example_excluded)
    print(f"Masked dataset summary written to: {new_path}")
    print("Preview (first 10 rows):")
    print(preview)


