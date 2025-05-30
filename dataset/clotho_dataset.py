import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import soundfile as sf
import faiss
import torchaudio
import time
import random


class ClothoDataset:
    """
    Clotho数据集类，用于加载和处理Clotho音频-文本对数据
    """
    def __init__(self, root_dir, type="full"):
        """
        初始化Clotho数据集
        
        Args:
            root_dir: 数据根目录，包含audio和captions的目录
            type: 数据集类型，可选"full"或"test"
                  "full"表示用development/validation/evaluation全部数据
                  "test"表示只用evaluation数据
        """
        self.root_dir = root_dir
        self.annotations = []
        self.audio_dirs = []
        self.split_names = []

        if type == "test":
            self.split_names = ["evaluation"]
        elif type == "full":
            self.split_names = ["development", "validation", "evaluation"]
        else:
            raise ValueError(f"Unsupported type: {type}")

        for split in self.split_names:
            caption_file = os.path.join(
                root_dir, f"clotho_captions_{split}.csv")
            audio_dir = os.path.join(root_dir, "clotho_audio", f"{split}")

            if not os.path.isfile(caption_file):
                raise ValueError(f"Caption file not found: {caption_file}")
            if not os.path.isdir(audio_dir):
                raise ValueError(f"Audio directory not found: {audio_dir}")

            df = pd.read_csv(caption_file)
            df["audio_dir"] = audio_dir  # 每行记录自己的音频路径
            self.annotations.append(df)

        self.annotations = pd.concat(self.annotations, ignore_index=True)

        # 生成text_ids和captions
        self.text_ids = []
        self.captions = []
        for _, row in self.annotations.iterrows():
            file_name = row["file_name"]
            for i in range(1, 6):
                caption_id = f"{file_name}_{i}"
                caption_text = row[f"caption_{i}"]
                self.text_ids.append(caption_id)
                self.captions.append(caption_text)

        self.audio_ids = self.annotations["file_name"].tolist()
        self.audio_dirs = self.annotations["audio_dir"].tolist()


class ClothoAudioDataset(Dataset):
    """
    Clotho音频数据集类，用于加载和处理音频数据
    """
    def __init__(self, clotho_dataset):
        """
        初始化Clotho音频数据集
        
        Args:
            clotho_dataset: ClothoDataset实例
        """
        self.audio_dirs = clotho_dataset.audio_dirs
        self.audio_ids = clotho_dataset.audio_ids

    def __len__(self):
        """返回数据集大小"""
        return len(self.audio_ids)

    def __getitem__(self, idx):
        """
        获取指定索引的音频数据
        
        Args:
            idx: 数据索引
            
        Returns:
            audio_id: 音频ID
            audio_path: 音频文件路径
        """
        audio_id = self.audio_ids[idx]
        audio_path = os.path.join(self.audio_dirs[idx], audio_id)
        return audio_id, audio_path


class ClothoTextDataset(Dataset):
    """
    Clotho文本数据集类，用于加载和处理文本描述数据
    """
    def __init__(self, clotho_dataset):
        """
        初始化Clotho文本数据集
        
        Args:
            clotho_dataset: ClothoDataset实例
        """
        self.text_ids = clotho_dataset.text_ids
        self.captions = clotho_dataset.captions

    def __len__(self):
        """返回数据集大小"""
        return len(self.text_ids)

    def __getitem__(self, idx):
        """
        获取指定索引的文本数据
        
        Args:
            idx: 数据索引
            
        Returns:
            text_id: 文本ID
            caption: 文本描述
        """
        return self.text_ids[idx], self.captions[idx]


# DataLoader
def clotho_get_audio_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取Clotho音频数据加载器
    
    Args:
        dataset: ClothoAudioDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        音频数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: list(zip(*batch))
    )


def clotho_get_text_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取Clotho文本数据加载器
    
    Args:
        dataset: ClothoTextDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        文本数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: list(zip(*batch))
    )


# qrels 和 runs 生成
def clotho_generate_qrels(dataset):
    """
    生成查询相关性文件
    每个音频有5条caption，每条caption和它对应的音频相关
    
    Args:
        dataset: ClothoDataset实例
        
    Returns:
        qrels: 查询相关性字典，格式为{文本ID: {音频ID: 相关性分数}}
    """
    qrels = {}
    for _, row in dataset.annotations.iterrows():
        file_name = row["file_name"]
        for i in range(1, 6):
            text_id = f"{file_name}_{i}"
            qrels[text_id] = {file_name: 1}
    return qrels


def clotho_generate_runs(model_wrapper, audio_loader, text_loader, device, target_sr=48000):
    """
    生成检索结果和计算编码时间
    
    Args:
        model_wrapper: 模型包装器
        audio_loader: 音频数据加载器
        text_loader: 文本数据加载器
        device: 计算设备
        target_sr: 目标采样率
        
    Returns:
        runs: 检索结果字典
        audio_encode_time: 总音频编码时间
        text_encode_time: 总文本编码时间
        avg_audio_time: 平均音频编码时间
        avg_txt_time: 平均文本编码时间
    """
    audio_embeds = []
    audio_ids = []
    audio_encode_time = 0.0

    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # 音频编码
    for ids, paths in tqdm(audio_loader, desc="Extracting audio features"):
        inputs = model_wrapper.preprocess_audio(paths, device=device)

        start_time = time.time()
        embeddings = model_wrapper.get_audio_features(inputs, device)
        end_time = time.time()

        audio_embeds.append(embeddings)
        audio_ids.extend(ids)
        audio_encode_time += (end_time - start_time)

    audio_embeds = np.vstack(audio_embeds)

    # 文本编码
    text_embeds = []
    text_ids = []
    text_encode_time = 0.0

    for ids, texts in tqdm(text_loader, desc="Extracting text features"):
        inputs = model_wrapper.preprocess_text(texts, device)
        start_time = time.time()
        embeddings = model_wrapper.get_text_features(inputs, device)
        end_time = time.time()

        text_embeds.append(embeddings)
        text_ids.extend(ids)
        text_encode_time += (end_time - start_time)

    text_embeds = np.vstack(text_embeds)

    # 使用Faiss进行检索
    dim = audio_embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(audio_embeds)

    D, I = index.search(text_embeds, 10)

    # 生成检索结果
    runs = {}
    for i, query_id in enumerate(text_ids):
        run = {}
        for rank, idx in enumerate(I[i]):
            doc_id = audio_ids[idx]
            run[doc_id] = float(D[i][rank])
        runs[query_id] = run

    # 计算平均编码时间
    num_audios = len(audio_loader.dataset)
    num_texts = len(text_loader.dataset)
    avg_audio_time = audio_encode_time / num_audios  # 每个音频的平均编码时间
    avg_txt_time = text_encode_time / num_texts    # 每段文本的平均编码时间

    return runs, audio_encode_time, text_encode_time, avg_audio_time, avg_txt_time
