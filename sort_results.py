import pandas as pd
import os

# 数据集类型映射
dataset_type_map = {
    "flickr30k": "image-text",
    "flickr8k": "image-text",
    "coco": "image-text",
    "msvd": "video-text",
    "msrvtt": "video-text",
    "audiocaps": "audio-text",
    "clotho": "audio-text",
    "t2retrieval": "text-only",
    "mmarcoretrieval": "text-only",
    "duretrieval": "text-only",
}

def get_dataset_type(dataset_name):
    return dataset_type_map.get(dataset_name.lower(), "unknown")

# 1. 获取 results 文件夹下所有 csv 文件
results_dir = os.path.join(os.path.dirname(__file__), 'results')
csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]

# 2. 读取所有 csv 文件并添加数据集类型列
all_dfs = []
for csv_file in csv_files:
    file_path = os.path.join(results_dir, csv_file)
    df = pd.read_csv(file_path)
    # 假设 dataset 列存在
    dataset_type = df['dataset'].map(get_dataset_type)
    # 插入到 dataset 前面一列
    insert_idx = df.columns.get_loc('dataset')
    df.insert(insert_idx, 'dataset_type', dataset_type)
    all_dfs.append(df)

# 3. 合并所有数据
merged_df = pd.concat(all_dfs, ignore_index=True)

# 4. 按照 dataset_type 和 dataset 排序
merged_df = merged_df.sort_values(by=['dataset_type', 'dataset'])

# 5. 输出整合后的 csv
merged_df.to_csv(os.path.join(results_dir, 'merged_sorted_results.csv'), index=False)
