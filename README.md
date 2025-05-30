# Baseline-Experiments

本项目为多模态数据库的基线实验集合，包含了多种主流多模态模型的实现与评测脚本，适用于图像、视频、文本等多模态数据的检索评测。

## 目录结构

```
.
├── dataset/                # 数据集相关文件
├── download/               # 数据集和模型下载目录
├── imagebind/              # ImageBind 模型相关代码
├── languagebind/           # LanguageBind 模型相关代码
├── VideoCLIP_XL/           # VideoCLIP-XL 模型相关代码
├── results/                # 实验结果保存目录
├── evaluation.py           # 评测脚本
├── main.py                 # 主程序入口
├── model.py                # 模型相关代码
├── requirements.txt        # Python 依赖包
├── run_model_tests.sh      # 批量运行模型测试的脚本
├── sort_results.py         # 结果排序脚本
└── README.md               # 项目说明文件
```

## 环境依赖

请先安装 Python 3.10 及以上版本。依赖包可通过如下命令安装：

```bash
pip install -r requirements.txt
```

## 快速开始

1. **下载数据集和模型**  
   将数据集和模型下载到 `download/datasets` 和 `download/models` 文件夹中。

2. **运行主程序**  
   以 `main.py` 为例，运行基线实验：

   ```bash
   python main.py --model_type bge --model_path ./download/models/bge-base-zh --dataset_name t2retrieval --dataset_type test
   ```

   你也可以使用 `run_model_tests.sh` 脚本批量测试不同模型：

   ```bash
   bash run_model_tests.sh
   ```

   该脚本支持以下参数：
   - `image`：运行文本-图像模型测试
   - `audio`：运行文本-音频模型测试
   - `video`：运行文本-视频模型测试
   - `text`：运行纯文本模型测试

## 主要文件说明

- `main.py`：主程序入口，负责加载数据、模型训练与测试。
- `model.py`：包含各类多模态模型的实现。
- `evaluation.py`：评测脚本，用于计算各类指标。
- `sort_results.py`：对实验结果进行排序和整理。
- `run_model_tests.sh`：批量运行不同模型的测试脚本。
- `requirements.txt`：项目依赖的 Python 包列表。
