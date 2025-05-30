import os
import pandas as pd
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import faiss
from tqdm import tqdm
import time
import numpy as np


class Flickr30kDataset:
    """
    Flickr30k数据集类，用于加载和处理Flickr30k图像-文本对数据
    """
    def __init__(self, image_dir, csv_path, type='full'):
        """
        初始化Flickr30k数据集
        
        Args:
            image_dir: 图像文件目录路径
            csv_path: 包含图像描述的CSV文件路径
            type: 数据集类型，可选"full"或"test"
        """
        self.image_dir = image_dir
        self.annotations = pd.read_csv(csv_path)
        if type == 'test':
            self.annotations = self.annotations[self.annotations["split"] == "test"]
        self.image_ids = self.annotations['img_id'].tolist()
        self.filenames = self.annotations['filename'].tolist()
        self.descriptions = self.annotations['raw'].apply(
            lambda x: json.loads(x)).tolist()
        self.sentids = self.annotations['sentids'].apply(
            lambda x: json.loads(x)).tolist()


class Flickr30kImageDataset(Dataset):
    """
    Flickr30k图像数据集类，用于加载和处理图像数据
    """
    def __init__(self, flickr_dataset):
        """
        初始化Flickr30k图像数据集
        
        Args:
            flickr_dataset: Flickr30kDataset实例
        """
        self.image_dir = flickr_dataset.image_dir
        self.image_ids = flickr_dataset.image_ids
        self.filenames = flickr_dataset.filenames
        self.transform = self.get_image_transform()

    def get_image_transform(self):
        """
        获取图像转换操作
        
        Returns:
            图像预处理转换操作
        """
        return T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor()
        ])

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        获取指定索引的图像数据
        
        Args:
            idx: 数据索引
            
        Returns:
            image_id: 图像ID
            image: 转换后的图像张量
        """
        image_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return self.image_ids[idx], image


class Flickr30kTextDataset(Dataset):
    """
    Flickr30k文本数据集类，用于加载和处理文本描述数据
    """
    def __init__(self, flickr_dataset):
        """
        初始化Flickr30k文本数据集
        
        Args:
            flickr_dataset: Flickr30kDataset实例
        """
        self.sentids = sum(flickr_dataset.sentids, [])
        self.captions = sum(flickr_dataset.descriptions, [])

    def __len__(self):
        """返回数据集大小"""
        return len(self.sentids)

    def __getitem__(self, idx):
        """
        获取指定索引的文本数据
        
        Args:
            idx: 数据索引
            
        Returns:
            sentid: 句子ID
            caption: 文本描述
        """
        return self.sentids[idx], self.captions[idx]


def flickr30k_get_image_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取Flickr30k图像数据加载器
    
    Args:
        dataset: Flickr30kImageDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        图像数据加载器
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def flickr30k_get_text_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取Flickr30k文本数据加载器
    
    Args:
        dataset: Flickr30kTextDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        文本数据加载器
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def flickr30k_generate_qrels(flickr_dataset):
    """
    生成查询相关性文件
    
    Args:
        flickr_dataset: Flickr30kDataset实例
        
    Returns:
        qrels: 查询相关性字典，格式为{句子ID: {图像ID: 相关性分数}}
    """
    qrels = {}
    for img_id, sentids in zip(flickr_dataset.image_ids, flickr_dataset.sentids):
        for sid in sentids:
            if sid not in qrels:
                qrels[str(sid)] = {}
            qrels[str(sid)][str(img_id)] = 1
    return qrels


def flickr30k_generate_runs(model_wrapper, image_loader, text_loader, device):
    """
    生成检索结果和计算编码时间
    
    Args:
        model_wrapper: 模型包装器
        image_loader: 图像数据加载器
        text_loader: 文本数据加载器
        device: 计算设备
        
    Returns:
        runs: 检索结果字典
        total_img_time: 总图像编码时间
        total_txt_time: 总文本编码时间
        avg_img_time: 平均图像编码时间
        avg_txt_time: 平均文本编码时间
    """
    runs = {}
    image_ids = []
    image_features = []
    image_times, text_times = [], []

    with torch.no_grad():
        # 图像编码
        for ids, images in tqdm(image_loader, desc="Extracting image features"):
            images = images.to(device)

            inputs = model_wrapper.preprocess_image(images, device)
            start = time.time()
            image_feats = model_wrapper.get_image_features(inputs)
            if isinstance(image_feats, torch.Tensor):
                image_feats = image_feats.cpu().numpy()
            image_times.append(time.time() - start)

            image_ids.extend([int(i) for i in ids])
            image_features.append(image_feats)

        image_features = np.vstack(image_features)
        # 特征归一化 (为了用内积等价于余弦相似度)
        image_features = image_features.astype('float32')
        faiss.normalize_L2(image_features)

        # 建立 Faiss 索引
        dim = image_features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(image_features)

    # 文本编码 + 检索
    text_features = []
    with torch.no_grad():
        for sent_ids, captions in tqdm(text_loader, desc="Extracting text features"):
            inputs = model_wrapper.preprocess_text(captions, device)

            start = time.time()
            text_feats = model_wrapper.get_text_features(inputs)
            if isinstance(text_feats, torch.Tensor):
                text_feats = text_feats.cpu().numpy()
            text_features.append(text_feats)
            text_times.append(time.time() - start)

    text_features = np.vstack(text_features)
    text_features = text_features.astype('float32')
    faiss.normalize_L2(text_features)

    # 文本检索
    sims, indices = index.search(text_features, k=10)
    for i, sid in enumerate(text_loader.dataset.sentids):
        runs[str(sid)] = {}
        for j in range(10):
            img_id = image_ids[indices[i][j]]
            sim = sims[i][j]
            runs[str(sid)][str(img_id)] = float(sim)

    total_img_time = sum(image_times)
    total_txt_time = sum(text_times)

    # 计算平均编码时间
    num_images = len(image_loader.dataset)
    num_texts = len(text_loader.dataset)
    avg_img_time = total_img_time / num_images  # 每张图片的平均编码时间
    avg_txt_time = total_txt_time / num_texts    # 每段文本的平均编码时间

    return runs, total_img_time, total_txt_time, avg_img_time, avg_txt_time
