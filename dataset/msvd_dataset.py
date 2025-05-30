import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import faiss
from tqdm import tqdm
import time


class MSVDDataset:
    """
    MSVD数据集类，用于加载和处理Microsoft Video Description数据集的视频-文本对数据
    """
    def __init__(self, root_dir):
        """
        初始化MSVD数据集
        
        Args:
            root_dir: 数据根目录，包含视频和字幕的目录
        """
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "YouTubeClips")
        self.caption_file = os.path.join(root_dir, "AllVideoDescriptions.txt")

        if not os.path.isfile(self.caption_file):
            raise ValueError(f"Caption file not found: {self.caption_file}")
        if not os.path.isdir(self.video_dir):
            raise ValueError(f"Video directory not found: {self.video_dir}")

        self.text_ids = []
        self.captions = []
        self.video_ids = []
        self.unique_video_dict = {}

        self._load_data()

    def _load_data(self):
        """
        读取并解析AllVideoDescriptions.txt文件，提取视频ID和对应的描述
        """
        with open(self.caption_file, "r") as f:
            lines = f.readlines()

        lines = lines[7:]  # 跳过头部信息

        text_counter = 0

        for line in lines:
            video_id, caption = line.split(" ", 1)
            video_path = os.path.join(self.video_dir, f"{video_id}.avi")

            if video_id not in self.unique_video_dict:
                self.unique_video_dict[video_id] = video_path

            text_id = f"{video_id}_{text_counter}"
            self.text_ids.append(text_id)
            self.captions.append(caption.strip())
            self.video_ids.append(video_id)

            text_counter += 1

    def __len__(self):
        """返回数据集大小"""
        return len(self.text_ids)


class MSVDVideoDataset(Dataset):
    """
    MSVD视频数据集类，用于加载和处理视频数据
    """
    def __init__(self, msdv_dataset):
        """
        初始化MSVD视频数据集
        
        Args:
            msdv_dataset: MSVDDataset实例
        """
        self.video_ids = list(msdv_dataset.unique_video_dict.keys())
        self.video_paths = list(msdv_dataset.unique_video_dict.values())

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


class MSVDTextDataset(Dataset):
    """
    MSVD文本数据集类，用于加载和处理文本描述数据
    """
    def __init__(self, msdv_dataset):
        """
        初始化MSVD文本数据集
        
        Args:
            msdv_dataset: MSVDDataset实例
        """
        self.text_ids = msdv_dataset.text_ids
        self.captions = msdv_dataset.captions

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


# DataLoader for MSVD
def msvd_get_video_dataloader(dataset, batch_size=8, num_workers=1):
    """
    获取MSVD视频数据加载器
    
    Args:
        dataset: MSVDVideoDataset实例
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


def msvd_get_text_dataloader(dataset, batch_size=8, num_workers=1):
    """
    获取MSVD文本数据加载器
    
    Args:
        dataset: MSVDTextDataset实例
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


# 视频帧转换函数
def video2image(video_path, frame_rate=1.0, size=224, num_frames=8):
    """
    将视频转换为图像帧序列
    
    Args:
        video_path: 视频文件路径
        frame_rate: 采样帧率
        size: 输出图像大小
        num_frames: 采样帧数量
        
    Returns:
        images: 转换后的图像帧张量 [num_frames, 3, size, size]
    """
    def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)

    cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        images = torch.zeros([num_frames, 3, size, size], dtype=torch.float32)
        print("ERROR: problem reading video file: ", video_path)
        return images
    
    total_duration = (frameCount + fps - 1) // fps
    start_sec, end_sec = 0, total_duration
    
    # 均匀选择指定数量的帧
    frames_idx = np.linspace(0, frameCount - 1, num_frames, dtype=int)
    images = torch.zeros([num_frames, 3, size, size], dtype=torch.float32)
    
    for i, idx in enumerate(frames_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # 如果读取失败，用零填充
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images[i] = preprocess(size, Image.fromarray(frame).convert("RGB"))
    
    cap.release()
    return images


def msvd_generate_qrels(dataset):
    """
    生成查询相关性文件
    每个文本描述和对应的视频是相关的
    
    Args:
        dataset: MSVDDataset实例
        
    Returns:
        qrels: 查询相关性字典，格式为{文本ID: {视频ID: 相关性分数}}
    """
    qrels = {}
    for text_id, video_id in zip(dataset.text_ids, dataset.video_ids):
        if text_id not in qrels:
            qrels[text_id] = {}
        qrels[text_id][video_id] = 1  # 1 表示该视频与描述相关
    return qrels


def msvd_generate_runs(model_wrapper, video_loader, text_loader, device):
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
