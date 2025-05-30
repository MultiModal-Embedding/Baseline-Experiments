import os
import pandas as pd
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import faiss
from tqdm import tqdm
import numpy as np
import time


class Flickr8kDataset:
    """
    Flickr8k数据集类，用于加载和处理Flickr8k图像-文本对数据
    """
    def __init__(self, parquet_dir, type="test"):
        """
        初始化Flickr8k数据集
        
        Args:
            parquet_dir: parquet文件目录路径
            type: 数据集类型，可选"full"或"test"
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
        self.image_ids = list(range(len(self.annotations)))

        self.captions_list = self.annotations[
            ['caption_0', 'caption_1', 'caption_2', 'caption_3', 'caption_4']
        ].values.tolist()


class Flickr8kImageDataset(Dataset):
    """
    Flickr8k图像数据集类，用于加载和处理图像数据
    """
    def __init__(self, flickr_dataset):
        """
        初始化Flickr8k图像数据集
        
        Args:
            flickr_dataset: Flickr8kDataset实例
        """
        self.image_ids = flickr_dataset.image_ids
        self.annotations = flickr_dataset.annotations
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
        img_bytes = self.annotations.iloc[idx]['image']['bytes']
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        image = self.transform(image)
        return self.image_ids[idx], image


class Flickr8kTextDataset(Dataset):
    """
    Flickr8k文本数据集类，用于加载和处理文本描述数据
    """
    def __init__(self, flickr_dataset):
        """
        初始化Flickr8k文本数据集
        
        Args:
            flickr_dataset: Flickr8kDataset实例
        """
        # 给每个caption一个唯一的id，形式"{image_id}_{0~4}"
        self.sentids = []
        self.captions = []
        for img_id, captions in zip(flickr_dataset.image_ids, flickr_dataset.captions_list):
            for i, caption in enumerate(captions):
                self.sentids.append(f"{img_id}_{i}")
                self.captions.append(caption)

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


def flickr8k_get_image_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取Flickr8k图像数据加载器
    
    Args:
        dataset: Flickr8kImageDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        图像数据加载器
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def flickr8k_get_text_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取Flickr8k文本数据加载器
    
    Args:
        dataset: Flickr8kTextDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        文本数据加载器
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def flickr8k_generate_qrels(flickr_dataset):
    """
    生成查询相关性文件
    
    Args:
        flickr_dataset: Flickr8kDataset实例
        
    Returns:
        qrels: 查询相关性字典，格式为{句子ID: {图像ID: 相关性分数}}
    """
    qrels = {}
    for img_id, captions in zip(flickr_dataset.image_ids, flickr_dataset.captions_list):
        for i in range(len(captions)):
            sid = f"{img_id}_{i}"
            qrels[sid] = {str(img_id): 1}
    return qrels


def flickr8k_generate_runs(model_wrapper, image_loader, text_loader, device):
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

        image_features = np.vstack(image_features).astype('float32')
        faiss.normalize_L2(image_features)

        # 建立Faiss索引
        dim = image_features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(image_features)

    # 文本编码 + 检索
    text_features = []
    sentids = text_loader.dataset.sentids

    with torch.no_grad():
        for batch_sentids, captions in tqdm(text_loader, desc="Extracting text features"):
            inputs = model_wrapper.preprocess_text(captions, device)
            start = time.time()
            text_feats = model_wrapper.get_text_features(inputs)
            if isinstance(text_feats, torch.Tensor):
                text_feats = text_feats.cpu().numpy()
            text_times.append(time.time() - start)

            text_features.append(text_feats)

    text_features = np.vstack(text_features).astype('float32')
    faiss.normalize_L2(text_features)

    # 文本检索
    sims, indices = index.search(text_features, k=10)

    for i, sid in enumerate(sentids):
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
