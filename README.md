<div align="center">


# Improving the detection of technical debt in Java source code with an enriched dataset

<!-- <p align="center">
  <img src="assets/logo.png" width="100px" alt="logo">
</p> -->

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Tesoro on HuggingFace datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-Tesoro-yellow?style=flat)](https://huggingface.co/datasets/Fsoft-AIC/the-vault-function) [![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat
)]() 

</div>

## Introduction
*Technical debt (TD)* arises when developers choose quick fixes over well-designed, long-term solutions. Self-Admitted Technical Debts (SATDs) are a type of TD where developers explicitly acknowledge shortcuts in code comments. Most existing approaches focus on analyzing these comments, often overlooking the source code itself. This study bridges the existing gap by developing the first dataset that associates SATD comments with their respective source code, and introduces a novel approach where the input consists solely of source code.

## Tesoro
We propose a novel dataset and construction pipeline (Figure below) to obtain informative samples for technical debt detection.

<img src="assets/pipeline.png" alt="logo">