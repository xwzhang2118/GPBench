## gwas-llm-judge 使用说明

本项目用于根据基因型/表型数据集以及历史实验经验，自动推荐合适的分析方法，并给出推荐理由。

### 安装依赖

在项目根目录 `/home/common/hwluo/project/gwas-llm-judge` 下执行：

```bash
cd /home/common/hwluo/project/gwas-llm-judge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 配置 LLM

在 `config/config.json` 中填写你的大模型配置，至少包含三个部分：

- **llm**：主对话模型（用于大部分推理）
- **codegen_llm**：偏代码/结构化生成的模型
- **multimodal_llm**：多模态模型（用于图像等）

一个简化示例（仅供参考，需根据你实际使用的模型服务修改）：

```json
{
  "llm": {
    "model": "gpt-4o-mini",
    "api_key": "YOUR_API_KEY",
    "base_url": "https://api.openai.com/v1"
  },
  "codegen_llm": {
    "model": "gpt-4o-mini",
    "api_key": "YOUR_API_KEY",
    "base_url": "https://api.openai.com/v1"
  },
  "multimodal_llm": {
    "model": "qwen-vl-max",
    "api_key": "YOUR_DASHSCOPE_API_KEY"
  }
}
```

### 数据准备

- **数据集目录**：例如 `datasets/Rapeseed`，需要至少包含：
  - `genetype.npz`
  - `phenotype.npz`
- **经验数据表**：项目自带 `experience/experience_origin.csv` 和 `experience/dataset_summary.csv`，用于相似数据集匹配与方法效果统计。

### 命令行调用方式

在虚拟环境已激活的前提下，在项目根目录执行：

```bash
python main.py \
  -d /home/common/hwluo/project/gwas-llm-judge/datasets/Rapeseed \
  -q "推荐这个数据集最合适的算法" \
  -m Rapeseed_GSTP013/FloweringTime \
  -o result.json
```

- **`-d / --dataset`**：数据集目录路径（必填）
- **`-q / --user-query`**：你的分析需求/问题描述（必填）
- **`-m / --mask`**：可选，指定一个 `species_phenotype`（如 `Rapeseed_GSTP013/FloweringTime`），在参考经验库中将其屏蔽，避免“泄露答案”
- **`-o / --output`**：可选，将结果保存为 JSON 文件路径；若不提供，则结果直接打印到终端

运行完成后，若使用了 `-o result.json`，你会在项目根目录看到一个 `result.json` 文件，内部包含：

- `similar_datasets`：相似数据集列表及整体理由
- `methods`：推荐方法列表及整体理由


