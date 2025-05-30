import os
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import faiss
from tqdm import tqdm
import time
import numpy as np
from io import BytesIO


class COCODataset:
    """
    COCO数据集类，用于加载和处理COCO图像-文本对数据
    """
    def __init__(self, parquet_dir, type='full'):
        """
        初始化COCO数据集
        
        Args:
            parquet_dir: parquet文件目录路径
            type: 数据集类型，可选"full"或"test"
        """
        self.annotations = self.load_parquet(parquet_dir, type)

        self.image_ids = self.annotations['cocoid'].tolist()
        self.captions = self.annotations['caption'].tolist()
        self.images = self.annotations['image'].tolist()

    def load_parquet(self, parquet_dir, type):
        """
        加载parquet格式的数据
        
        Args:
            parquet_dir: parquet文件目录
            type: 数据集类型
            
        Returns:
            合并后的DataFrame
        """
        if type == 'test':
            target_prefixes = ['test']
        elif type == 'full':
            target_prefixes = ['train', 'validation', 'test']
        else:
            raise ValueError(f"Unsupported type: {type}")

        files = [f for f in os.listdir(parquet_dir)
                 if any(f.startswith(prefix) for prefix in target_prefixes)]
        if not files:
            raise ValueError(f"No parquet files found for type: {type}")

        dfs = []
        for f in files:
            table = pq.read_table(os.path.join(parquet_dir, f))
            df = table.to_pandas()
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)


class COCOImageDataset(Dataset):
    """
    COCO图像数据集类，用于加载和处理图像数据
    """
    def __init__(self, coco_dataset):
        """
        初始化COCO图像数据集
        
        Args:
            coco_dataset: COCODataset实例
        """
        self.image_ids = coco_dataset.image_ids
        self.images = coco_dataset.images
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
        image_dict = self.images[idx]
        img_bytes = image_dict['bytes']
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        image = self.transform(image)
        return self.image_ids[idx], image


class COCOTextDataset(Dataset):
    """
    COCO文本数据集类，用于加载和处理文本描述数据
    """
    def __init__(self, coco_dataset):
        """
        初始化COCO文本数据集
        
        Args:
            coco_dataset: COCODataset实例
        """
        self.image_ids = coco_dataset.image_ids
        self.captions = coco_dataset.captions

    def __len__(self):
        """返回数据集大小"""
        return len(self.captions)

    def __getitem__(self, idx):
        """
        获取指定索引的文本数据
        
        Args:
            idx: 数据索引
            
        Returns:
            image_id: 对应的图像ID
            caption: 文本描述
        """
        return self.image_ids[idx], self.captions[idx]


def coco_get_image_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取COCO图像数据加载器
    
    Args:
        dataset: COCOImageDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        图像数据加载器
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def coco_get_text_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取COCO文本数据加载器
    
    Args:
        dataset: COCOTextDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        文本数据加载器
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def coco_generate_qrels(coco_dataset):
    """
    生成查询相关性文件
    COCO每张图片有多条caption，生成qrels映射
    
    Args:
        coco_dataset: COCODataset实例
        
    Returns:
        qrels: 查询相关性字典，格式为{caption: {图像ID: 相关性分数}}
    """
    qrels = {}
    for img_id, caption in zip(coco_dataset.image_ids, coco_dataset.captions):
        if str(caption) not in qrels:
            qrels[str(caption)] = {}
        qrels[str(caption)][str(img_id)] = 1
    return qrels


def coco_generate_runs(model_wrapper, image_loader, text_loader, device):
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
    with torch.no_grad():
        for img_ids, captions in tqdm(text_loader, desc="Extracting text features"):
            inputs = model_wrapper.preprocess_text(captions, device)

            start = time.time()
            text_feats = model_wrapper.get_text_features(inputs)
            if isinstance(text_feats, torch.Tensor):
                text_feats = text_feats.cpu().numpy()
            text_features.append(text_feats)
            text_times.append(time.time() - start)

    text_features = np.vstack(text_features).astype('float32')
    faiss.normalize_L2(text_features)

    # 文本检索
    sims, indices = index.search(text_features, k=10)
    for i, caption in enumerate(text_loader.dataset.captions):
        runs[str(caption)] = {}
        for j in range(10):
            img_id = image_ids[indices[i][j]]
            sim = sims[i][j]
            runs[str(caption)][str(img_id)] = float(sim)

    total_img_time = sum(image_times)
    total_txt_time = sum(text_times)

    # 计算平均编码时间
    num_images = len(image_loader.dataset)
    num_texts = len(text_loader.dataset)
    avg_img_time = total_img_time / num_images  # 每张图片的平均编码时间
    avg_txt_time = total_txt_time / num_texts    # 每段文本的平均编码时间

    return runs, total_img_time, total_txt_time, avg_img_time, avg_txt_time
