import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import faiss
import time

# Dataset
class AudioCapsDataset:
    """
    AudioCaps数据集类，用于加载和处理音频-文本对数据
    """
    def __init__(self, parquet_dir, type="test"):
        """
        初始化AudioCaps数据集
        
        Args:
            parquet_dir: parquet文件目录路径
            type: 数据集类型，可选"test"或"full"
        """
        if type == "full":
            files = [f for f in os.listdir(
                parquet_dir) if f.endswith(".parquet")]
        else:
            files = [f for f in os.listdir(parquet_dir) if f.startswith(type)]

        if not files:
            raise ValueError(f"No files found for type {type}")

        df_list = [pd.read_parquet(os.path.join(parquet_dir, f))
                   for f in files]
        self.annotations = pd.concat(df_list, ignore_index=True)
        self.audio_ids = self.annotations['youtube_id'].tolist()
        self.text_ids = [str(i)
                         for i in self.annotations['audiocap_id'].tolist()]
        self.captions = self.annotations['caption'].tolist()
        self.audio_bytes = self.annotations['audio'].tolist()

        # 创建唯一音频字典
        self.unique_audio_dict = {}
        for audio_id, audio_data in zip(self.audio_ids, self.audio_bytes):
            if audio_id not in self.unique_audio_dict:
                self.unique_audio_dict[audio_id] = audio_data


class AudioCapsAudioDataset(Dataset):
    """
    AudioCaps音频数据集类，用于加载和处理音频数据
    """
    def __init__(self, audiocaps_dataset):
        """
        初始化AudioCaps音频数据集
        
        Args:
            audiocaps_dataset: AudioCapsDataset实例
        """
        self.audio_ids = list(audiocaps_dataset.unique_audio_dict.keys())
        self.audio_bytes = list(audiocaps_dataset.unique_audio_dict.values())

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
            audio_bytes: 音频字节数据
        """
        return self.audio_ids[idx], self.audio_bytes[idx]


class AudioCapsTextDataset(Dataset):
    """
    AudioCaps文本数据集类，用于加载和处理文本描述数据
    """
    def __init__(self, audiocaps_dataset):
        """
        初始化AudioCaps文本数据集
        
        Args:
            audiocaps_dataset: AudioCapsDataset实例
        """
        self.text_ids = audiocaps_dataset.text_ids
        self.captions = audiocaps_dataset.captions

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
def audiocaps_get_audio_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取AudioCaps音频数据加载器
    
    Args:
        dataset: AudioCapsAudioDataset实例
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


def audiocaps_get_text_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取AudioCaps文本数据加载器
    
    Args:
        dataset: AudioCapsTextDataset实例
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
def audiocaps_generate_qrels(dataset):
    """
    生成查询相关性文件
    每个音频只和它自己的 caption 是相关的
    
    Args:
        dataset: AudioCapsDataset实例
        
    Returns:
        qrels: 查询相关性字典，格式为{文本ID: {音频ID: 相关性分数}}
    """
    qrels = {}
    for audio_id, text_id in zip(dataset.audio_ids, dataset.text_ids):
        qrels[text_id] = {audio_id: 1}
    return qrels


def audiocaps_generate_runs(model_wrapper, audio_loader, text_loader, device):
    """
    生成检索结果和计算编码时间
    
    Args:
        model_wrapper: 模型包装器
        audio_loader: 音频数据加载器
        text_loader: 文本数据加载器
        device: 计算设备
        
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

    # 音频编码
    for ids, audios in tqdm(audio_loader, desc="Extracting audio features"):
        # audios 是原始字节
        inputs = model_wrapper.preprocess_audio(audios, device=device)

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

    # 建立索引和搜索
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
