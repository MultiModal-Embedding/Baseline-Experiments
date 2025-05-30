import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import json
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import faiss
from tqdm import tqdm
import time


class MSRVTTDataset:
    """
    MSR-VTT数据集类，用于加载和处理MSR-VTT视频-文本对数据
    """
    def __init__(self, root_dir, type='full'):
        """
        初始化MSR-VTT数据集
        
        Args:
            root_dir: 数据根目录，包含视频和注释文件的目录
            type: 数据集类型，可选"full"或"test"，决定使用哪些数据文件
        """
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "video")
        self.type = type

        # 根据type决定要读取的文件
        self.annotation_files = []
        if type == 'test':
            self.annotation_files.append(os.path.join(root_dir, "msrvtt_test_1k.json"))
        else:  # full
            self.annotation_files.extend([
                os.path.join(root_dir, "msrvtt_test_1k.json"),
                os.path.join(root_dir, "msrvtt_train_9k.json")
            ])

        # 检查文件是否存在
        for file in self.annotation_files:
            if not os.path.isfile(file):
                raise ValueError(f"Annotation file not found: {file}")
        if not os.path.isdir(self.video_dir):
            raise ValueError(f"Video directory not found: {self.video_dir}")

        self.text_ids = []
        self.captions = []
        self.video_ids = []
        self.unique_video_dict = {}

        self._load_data()

    def _load_data(self):
        """
        读取并解析注释文件，提取视频ID和对应的描述
        test json: caption是字符串
        train json: caption是字符串列表
        """
        text_counter = 0
        
        for annotation_file in self.annotation_files:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # 处理每个视频及其对应的描述
            for video_item in data:
                video_id = str(video_item['video_id'])
                captions = video_item['caption']
                
                # 构建视频路径
                video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
                
                # 如果是新的视频ID且视频文件存在，添加到字典中
                if video_id not in self.unique_video_dict and os.path.exists(video_path):
                    self.unique_video_dict[video_id] = video_path
                
                # 只有当视频文件存在时才添加caption
                if video_id in self.unique_video_dict:
                    # 根据caption类型进行不同处理
                    if isinstance(captions, str):
                        # test json格式，caption是字符串
                        text_id = f"{video_id}_0"  # 每个视频只有一个caption
                        self.text_ids.append(text_id)
                        self.captions.append(captions.strip())
                        self.video_ids.append(video_id)
                        text_counter += 1
                    else:
                        # train json格式，caption是列表
                        for caption_idx, caption in enumerate(captions):
                            text_id = f"{video_id}_{caption_idx}"
                            self.text_ids.append(text_id)
                            self.captions.append(caption.strip())
                            self.video_ids.append(video_id)
                        text_counter += len(captions)

    def __len__(self):
        """返回数据集大小"""
        return len(self.text_ids)


class MSRVTTVideoDataset(Dataset):
    """
    MSR-VTT视频数据集类，用于加载和处理视频数据
    """
    def __init__(self, msrvtt_dataset):
        """
        初始化MSR-VTT视频数据集
        
        Args:
            msrvtt_dataset: MSRVTTDataset实例
        """
        self.video_ids = list(msrvtt_dataset.unique_video_dict.keys())
        self.video_paths = list(msrvtt_dataset.unique_video_dict.values())

    def __len__(self):
        """返回数据集大小"""
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        获取指定索引的视频数据
        
        Args:
            idx: 数据索引
            
        Returns:
            video_id: 视频ID
            video_path: 视频文件路径
        """
        video_id = self.video_ids[idx]
        video_path = self.video_paths[idx]
        return video_id, video_path


class MSRVTTTextDataset(Dataset):
    """
    MSR-VTT文本数据集类，用于加载和处理文本描述数据
    """
    def __init__(self, msrvtt_dataset):
        """
        初始化MSR-VTT文本数据集
        
        Args:
            msrvtt_dataset: MSRVTTDataset实例
        """
        self.text_ids = msrvtt_dataset.text_ids
        self.captions = msrvtt_dataset.captions

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


def msrvtt_get_video_dataloader(dataset, batch_size=8, num_workers=1):
    """
    获取MSR-VTT视频数据加载器
    
    Args:
        dataset: MSRVTTVideoDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        视频数据加载器
    """
    def collate_fn(batch):
        video_ids, video_paths = zip(*batch)
        return list(video_ids), list(video_paths)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def msrvtt_get_text_dataloader(dataset, batch_size=8, num_workers=1):
    """
    获取MSR-VTT文本数据加载器
    
    Args:
        dataset: MSRVTTTextDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        文本数据加载器
    """
    def collate_fn(batch):
        text_ids, captions = zip(*batch)
        return text_ids, captions

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def msrvtt_generate_qrels(dataset):
    """
    生成查询相关性文件
    每个文本描述和对应的视频是相关的
    
    Args:
        dataset: MSRVTTDataset实例
        
    Returns:
        qrels: 查询相关性字典，格式为{文本ID: {视频ID: 相关性分数}}
    """
    qrels = {}
    for text_id, video_id in zip(dataset.text_ids, dataset.video_ids):
        if text_id not in qrels:
            qrels[text_id] = {}
        qrels[text_id][video_id] = 1  # 1 表示该视频与描述相关
    return qrels


def msrvtt_generate_runs(model_wrapper, video_loader, text_loader, device):
    """
    生成检索结果和计算编码时间
    
    Args:
        model_wrapper: 模型包装器
        video_loader: 视频数据加载器
        text_loader: 文本数据加载器
        device: 计算设备
        
    Returns:
        runs: 检索结果字典
        video_encode_time: 总视频编码时间
        text_encode_time: 总文本编码时间
        avg_video_time: 平均视频编码时间
        avg_txt_time: 平均文本编码时间
    """
    video_embeds = []
    video_ids = []
    video_encode_time = 0.0

    # 提取视频特征
    for video_batch_ids, video_batch_paths in tqdm(video_loader, desc="Extracting video features"):
        # video_batch_paths: list of video_path
        video_batch_frames = model_wrapper.preprocess_video(video_batch_paths)
        start_time = time.time()
        embeddings = model_wrapper.get_video_features(video_batch_frames, device)
        video_encode_time += time.time() - start_time
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        video_embeds.append(embeddings)
        video_ids.extend(video_batch_ids)

    video_embeds = np.vstack(video_embeds)

    # 提取文本特征
    text_embeds = []
    text_ids = []
    text_encode_time = 0.0

    for text_batch_ids, text_batch in tqdm(text_loader, desc="Extracting text features"):
        inputs = model_wrapper.preprocess_text(text_batch, device)
        start_time = time.time()
        embeddings = model_wrapper.get_text_features(inputs, device)
        text_encode_time += time.time() - start_time
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        text_embeds.append(embeddings)
        text_ids.extend(text_batch_ids)

    text_embeds = np.vstack(text_embeds)

    # 使用Faiss进行检索
    dim = video_embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(video_embeds)

    D, I = index.search(text_embeds, 10)

    # 生成检索结果
    runs = {}
    for i, query_id in enumerate(text_ids):
        run = {}
        for rank, idx in enumerate(I[i]):
            doc_id = video_ids[idx]
            run[doc_id] = float(D[i][rank])
        runs[query_id] = run

    # 计算平均编码时间
    num_videos = len(video_loader.dataset)
    num_texts = len(text_loader.dataset)
    avg_video_time = video_encode_time / num_videos  # 每个视频的平均编码时间
    avg_txt_time = text_encode_time / num_texts    # 每段文本的平均编码时间

    return runs, video_encode_time, text_encode_time, avg_video_time, avg_txt_time