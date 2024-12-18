<div align="center">


# Improving the detection of technical debt in Java source code with an enriched dataset

<!-- <p align="center">
  <img src="assets/logo.png" width="300px" alt="logo">
</p> -->

[![License: MIT](https://custom-icon-badges.demolab.com/badge/License-MIT-green.svg?logo=law)](https://opensource.org/licenses/MIT) [![Tesoro on HuggingFace datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-Tesoro-yellow?style=flat)](https://huggingface.co/datasets/NamCyan/tesoro-code) [![Python](https://custom-icon-badges.demolab.com/badge/Python-3.10+-blue?style=flat&logo=python)]() [![Paper](https://img.shields.io/badge/arxir-2411.05457-red?style=flat)](https://arxiv.org/abs/2411.05457) 

</div>

## Table of content
- [Introduction](#introduction)
- [Tesoro](#tesoro)
  - [Data Usage](#data-usage)
  - [Data Structure](#data-structure)
  - [Data for Experiments](#data-for-experiments)
- [Experiment Replication](#experiment-replication)
- [Leaderboard](#leaderboard)
- [Reference](#reference)
- [License](#license)

___________


# Introduction
*Technical debt (TD)* arises when developers choose quick fixes over well-designed, long-term solutions. Self-Admitted Technical Debts (SATDs) are a type of TD where developers explicitly acknowledge shortcuts in code comments. Most existing approaches focus on analyzing these comments, often overlooking the source code itself. This study bridges the existing gap by developing the first dataset that associates SATD comments with their respective source code, and introduces a novel approach where the input consists solely of source code.

*All resources (datasets and models) can be found at [Tesoro Hub](https://huggingface.co/collections/NamCyan/tesoro-671ba96dd7c96bdc4aea22cd)* 🎉.

# Tesoro
We propose a novel dataset and construction pipeline (Fig. 1) to obtain informative samples for technical debt detection.

<img src="assets/pipeline.png" alt="logo">

## Data Usage
$\text{Tesoro}$ contains two datasets:

- $\text{Tesoro}_{comment}$: comments serve as the input source, to support SATD-related tasks. Source code can be used as additional context.

- $\text{Tesoro}_{code}$: supports detecting technical debt in source code without relying on natural language comments.

<br>

**Dataset on Huggingface:** We publish [`tesoro-comment`](https://huggingface.co/datasets/NamCyan/tesoro-comment) and [`tesoro-code`](https://huggingface.co/datasets/NamCyan/tesoro-code) on Huggingface Dataset Hub 🤗


```python
from datasets import load_dataset

# Load Tesoro comment
dataset = load_dataset("NamCyan/tesoro-comment")

# Load Tesoro code
dataset = load_dataset("NamCyan/tesoro-code")
```

**Dataset on Github**: Tesoro is also available in this repository at [data/tesoro](data/tesoro/).

## Data Structure
- `tesoro-comment`
```json
{
    "id": "function id in the dataset",
    "comment_id": "comment id of the function",
    "comment": "comment text",
    "classification": "technical debt types (DESIGN | IMPLEMENTATION | DEFECT | DOCUMENTATION | TEST | NONSATD)",
    "code": "full fucntion context",
    "code_context_2": "2 lines code context",
    "code_context_10": "10 lines code context",
    "code_context_20": "20 lines code context",
    "repo": "Repository that contains this source"
}
```

- `tesoro-code`
```json
{
    "id": "function id in the dataset",
    "original_code": "raw function",
    "code_wo_comment": "original code without comment",
    "cleancode": "normalized version of code (lowercase, remove newline \n)",
    "label": "binary list corresponding to 4 TD types (DESIGN, IMPLEMENATION, DEFECT, TEST)",
    "repo": "Repository that contains this source"
}
```

## Data for Experiments

The data prepared for training the SATD detector, performing k-fold evaluation, and answering the research questions is detailed in [Data for Experiments](data/README.md).


# Experiment Replication

We answer three research questions:

- **RQ1:** *Do the manually classified comments contribute to an improvement in the detection of SATD?*

- **RQ2:** *Does the inclusion of source code help to enhance the detection of technical debt?*

- **RQ3:** *What is the accuracy of different pre-trained models when detecting TD solely from source code?*

All results can be found [here](results). To reproduce the results of our experiments, see [Training](training/README.md) for more details.

# Leaderboard

| Model   <img width="400" height="1">     | Model size <img width="100" height="1">  | EM  <img width="100" height="1">           | F1  <img width="100" height="1">               |
|:-------------|:-----------|:------------------|:------------------|
| **Encoder-based PLMs** |
| [CodeBERT](https://huggingface.co/microsoft/codebert-base)     | 125M       | 38.28             | 43.47             |
| [UniXCoder](https://huggingface.co/microsoft/unixcoder-base)    | 125M       | 38.12             | 42.58             |
| [GraphCodeBERT](https://huggingface.co/microsoft/graphcodebert-base)| 125M       | *39.38*          | *44.21*           |
| [RoBERTa](https://huggingface.co/FacebookAI/roberta-base)      | 125M       | 35.37             | 38.22             |
| [ALBERT](https://huggingface.co/albert/albert-base-v2)       | 11.8M      | 39.32             | 41.99             |
| **Encoder-Decoder-based PLMs** |
| [PLBART](https://huggingface.co/uclanlp/plbart-base)       | 140M       | 36.85             | 39.90             |
| [Codet5](https://huggingface.co/Salesforce/codet5-base)       | 220M       | 32.66             | 35.41             |
| [CodeT5+](https://huggingface.co/Salesforce/codet5p-220m)      | 220M       | 37.91             | 41.96             |
| **Decoder-based PLMs (LLMs)** |
| [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama_v1.1_math_code)    | 1.03B      | 37.05             | 40.05             |
| [DeepSeek-Coder](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) | 1.28B    | **42.52**         | **46.19**         |
| [OpenCodeInterpreter](https://huggingface.co/m-a-p/OpenCodeInterpreter-DS-1.3B)       | 1.35B             | 38.16             | 41.76             |
| [phi-2](https://huggingface.co/microsoft/phi-2)        | 2.78B      | 37.92             | 41.57             |
| [starcoder2](https://huggingface.co/bigcode/starcoder2-3b)   | 3.03B      | 35.37             | 41.77             |
| [CodeLlama](https://huggingface.co/codellama/CodeLlama-7b-hf)    | 6.74B      | 34.14             | 38.16             |
| [Magicoder](https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B)    | 6.74B      | 39.14             | 42.49             |

# Reference

If you're using Tesoro, please cite using this BibTeX:

```bibtex
@article{nam2024tesoro,
  title={Improving the detection of technical debt in Java source code with an enriched dataset},
  author={Hai, Nam Le and Bui, Anh M. T. Bui and Nguyen, Phuong T. and Ruscio, Davide Di and Kazman, Rick},
  journal={},
  year={2024}
}
```

# License
[MIT License](LICENSE)
