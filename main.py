import os
import torch
import pandas as pd
import argparse
from model import (
    CLIPModelWrapper, CLAPModelWrapper, CLIP4CLIPModelWrapper, 
    BLIPModelWrapper, ALIGNModelWrapper, LanguageBindModelWrapper,
    ImageBindModelWrapper, GTEModelWrapper, BGEModelWrapper, M3EModelWrapper, XCLIPModelWrapper, VideoCLIPXLModelWrapper
)
from dataset.base_dataset import get_dataset
from evaluation import evaluate_ranking


def main(args):
    print("\n=== 运行配置 ===")
    print(f"模型类型: {args.model_type}")
    print(f"模型路径: {args.model_path}")
    print(f"数据集名称: {args.dataset_name}")
    print(f"数据集类型: {args.dataset_type}")
    print(f"运行模式: {args.modes}")
    print(f"结果保存路径: {args.save_path}")
    print("=" * 20 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载指定数据集
    dataset_bundle = get_dataset(
        args.dataset_name,
        type=args.dataset_type
    )

    # 获取数据集相关组件
    loader_key = dataset_bundle['primary_loader_key']  # 获取主要加载器的键名
    primary_loader = dataset_bundle[loader_key]  # 获取主要数据加载器（图像/音频/视频等）
    text_loader = dataset_bundle["text_loader"] if "text_loader" in dataset_bundle else dataset_bundle["query_loader"]  # 获取文本数据加载器
    qrels = dataset_bundle["qrels"]  # 获取查询相关性评分
    generate_runs = dataset_bundle["generate_runs"]  # 获取生成运行函数

    modes = args.modes.split(",")
    results = []

    # 对每个运行模式进行评估
    for mode in modes:
        print(f"\n=== Mode: {mode} ===")

        # 根据模型类型初始化对应的模型包装器
        if args.model_type == "clip":
            model_wrapper_q = CLIPModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "clap":
            model_wrapper_q = CLAPModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "clip4clip":
            model_wrapper_q = CLIP4CLIPModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "xclip":
            model_wrapper_q = XCLIPModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "blip":
            model_wrapper_q = BLIPModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "align":
            model_wrapper_q = ALIGNModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "languagebind":
            model_wrapper_q = LanguageBindModelWrapper(device=device)
        elif args.model_type == "imagebind":
            model_wrapper_q = ImageBindModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "gte":
            model_wrapper_q = GTEModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "bge":
            model_wrapper_q = BGEModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "m3e":
            model_wrapper_q = M3EModelWrapper(args.model_path, mode=mode)
        elif args.model_type == "videoclipxl":
            model_wrapper_q = VideoCLIPXLModelWrapper(args.model_path, mode=mode)

        # 生成检索运行结果并记录时间
        runs, total_primary_time, total_text_time, avg_primary_time, avg_text_time = generate_runs(
            model_wrapper_q, primary_loader, text_loader, device
        )

        # 评估检索性能
        metrics = evaluate_ranking(qrels, runs)

        # 收集结果
        result = {
            "model": os.path.basename(args.model_path),  # 模型名称
            "dataset": args.dataset_name,  # 数据集名称
            "type": args.dataset_type,  # 数据集类型
            "mode": mode,  # 运行模式
            "total_primary_encoding_time(s)": round(total_primary_time, 4),  # 主要模态编码总时间(秒)
            "total_text_encoding_time(s)": round(total_text_time, 4),  # 文本编码总时间(秒)
            "avg_primary_encoding_time(ms)": round(avg_primary_time * 1000, 4),  # 平均主要模态编码时间(毫秒)
            "avg_text_encoding_time(ms)": round(avg_text_time * 1000, 4),  # 平均文本编码时间(毫秒)
            "num_queries": len(runs),  # 查询数量
        }
        result.update({k: round(v, 4) for k, v in metrics.items()})
        results.append(result)

    df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(df)

    write_header = not os.path.isfile(args.save_path)
    df.to_csv(args.save_path, mode="a", index=False, header=write_header)
    print(f"\nResults saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 模型类型
    parser.add_argument("--model_type", type=str, 
                       choices=["clip", "clap", "clip4clip", "xclip", "videoclipxl", "blip", "align", "languagebind", "imagebind", "gte", "bge", "m3e"], 
                       default="m3e",
                       help="Type of model: clip, clap, clip4clip, xclip, videoclipxl, blip, align, languagebind, imagebind, gte, bge, or m3e")
    # 模型路径
    parser.add_argument("--model_path", type=str, default="./download/models/m3e-base",
                       help="Path to the model directory")
    # 数据集名称
    parser.add_argument("--dataset_name", type=str, default="t2retrieval",
                       help="Name of the dataset")
    # 数据集类型
    parser.add_argument("--dataset_type", type=str, default="test",
                       help="Dataset split type: test, full")
    # 运行模式
    parser.add_argument("--modes", type=str, default="fp32",
                       help="Comma-separated list of modes: fp32, fp16, int8, int4")
    # 结果保存路径
    parser.add_argument("--save_path", type=str, default="results.csv",
                       help="CSV file to save results")

    args = parser.parse_args()
    main(args)
