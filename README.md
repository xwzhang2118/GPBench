# GPBench

GPBench 是一个面向基因组预测（genomic prediction）的对比基准工具箱。该仓库重新实现并整合了众多常用方法，包含经典线性统计法与机器学习/深度学习方法：rrBLUP、GBLUP、BayesA/B/C、SVR、Random Forest、XGBoost、LightGBM、DeepGS、DL_GWAS、DNNGP、SoyDNGP、DeepCCR、EIR、Cropformer、GEFormer、CropARNet 等。

## 主要特性
- 提供多种基因组预测方法的实现与可复现实验流程
- 支持 GPU 加速的深度学习方法（使用 PyTorch）
- 统一的数据读取与 10 折交叉验证流程
- 输出标准化的评估指标（PCC、MAE、MSE、R2）与每折预测结果

## 目录结构（重要部分）
- `data/`：示例/真实数据集目录，每个物种/数据集为一个子文件夹（例如 `data/Cotton/`），其中包含：
	- `genetype.npz`：基因型矩阵（通常为 numpy 保存的数组）
	- `phenotype.npz`：表型数据（包含表型矩阵与表型名称）
- `method/`：各方法实现子目录（每个方法通常含主运行脚本与超参数/辅助脚本）
- `result/`：默认的实验结果输出目录
- `environment.yml`：用于创建 conda 环境的依赖文件（推荐使用）

## 环境安装（推荐：conda）
仓库中已有 `environment.yml`，建议直接用它创建并激活 conda 环境：

```bash
# 在有 conda 的机器上：
conda env create -f environment.yml
conda activate Benchmark
```

说明：
- `environment.yml` 已包含大部分依赖（含 CUDA / cuDNN 相关包与 pip 列表），适用于带 GPU 的环境（文件内使用的是 CUDA 11.8 与对应的 RAPIDS/torch/cupy 版本）。
- 请确保目标机器已安装匹配的 NVIDIA 驱动（与 CUDA 11.8/12 系列兼容）。
- 如果无法直接用该 environment 文件，可以在已有 Python 环境中按需安装主要依赖：

```bash
pip install -U numpy pandas scikit-learn torch torchvision optuna psutil xgboost lightgbm
```

（注意：上述为简化安装，某些包在 GPU 环境或特定平台需要额外配置）

## 数据格式与准备
- 每个 species 文件夹应包含 `genetype.npz` 与 `phenotype.npz`。
- `genetype.npz` 中通常保存一个二维数组（样本数 × SNP 数）。
- `phenotype.npz` 通常包含两个数组：表型矩阵（样本数 × 表型数）与表型名称列表。

快速查看某个数据集的表型名称（以 `Cotton` 为例）：

```bash
python - <<'PY'
import numpy as np
obj = np.load('data/Cotton/phenotype.npz')
print(obj['arr_1'])
PY
```

## 快速开始（以某个方法为例）
大多数方法在 `method/<Method>/` 下有一个主脚本，脚本通常接收参数：`--methods`、`--species`、`--phe`、`--data_dir`、`--result_dir` 等。示例如下：

```bash
# 1) 激活环境
conda activate Benchmark

# 2) 以 DeepCCR 为例运行单个表型（注意：--species 后面包含目录名末尾的斜杠）
python method/DeepCCR/DeepCCR.py \
	--methods DeepCCR/ \
	--species Cotton/ \
	--phe <PHENOTYPE_NAME> \
	--data_dir data/ \
	--result_dir result/
```

常见可选参数（不同脚本可能略有不同）：
- `--epoch`：训练轮数（默认示例脚本里通常为 1000）
- `--batch_size`：批大小
- `--lr`：学习率
- `--patience`：早停等待轮数

可以在对应方法目录下查看该脚本的 `argparse` 帮助，得到完整参数：

```bash
python method/DeepCCR/DeepCCR.py -h
```

## 输出说明
- 每次运行会在 `result/` 下创建以方法/物种/表型命名的目录，例如 `result/DeepCCR/Cotton/<PHENO>/`。
- 每折的预测结果通常保存为 `fold{n}.csv`，其中包含 `Y_test` 与 `Y_pred` 列。
- 脚本运行结束会打印或保存平均评估指标：PCC（Pearson correlation coefficient）、MAE、MSE、R2 及运行时间与内存/GPU 使用情况。

## 完整数据集链接
- [Species](https://drive.google.com/file/d/1-y71y9-y6-y9-)：包含 16 个物种的基因型与表型数据。

## 运行提示与常见问题
- 若使用 GPU，请确保 `conda activate Benchmark` 后 CUDA 驱动可用，且 `torch.cuda.is_available()` 返回 True。
- 对于内存或 GPU 显存不足，可尝试减小 `--batch_size` 或在脚本中禁用某些并行设置。
- 若使用 CPU 执行（无 GPU），某些方法（RAPIDS 或 GPU 专用实现）可能不可用或需要替代实现。

## 贡献 & 联系
- 欢迎提交 issue 与 PR。请在 PR 中说明更改目的与测试方式。
- 联系方式：在仓库 `Issues` 中留言或联系仓库所有者（GitHub 用户名：`xwzhang2118`）。





