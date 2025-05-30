import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import faiss
from tqdm import tqdm
import numpy as np
import time


class MMarcoRetrievalDataset:
    """
    MMarcoRetrieval数据集类，用于加载和处理MS MARCO检索数据集的文本数据
    """
    def __init__(self, parquet_dir, type="test"):
        """
        初始化MMarcoRetrieval数据集
        
        Args:
            parquet_dir: parquet文件目录路径
            type: 数据集类型，可选"full"或"test"
        """
        # 加载语料库数据
        corpus_dir = os.path.join(parquet_dir, "data")
        corpus_files = [f for f in os.listdir(corpus_dir) if f.startswith("corpus-")]
        corpus_dfs = [pd.read_parquet(os.path.join(corpus_dir, f)) for f in corpus_files]
        self.corpus = pd.concat(corpus_dfs, ignore_index=True)

        # 加载查询数据
        query_files = [f for f in os.listdir(corpus_dir) if f.startswith("queries-")]
        query_dfs = [pd.read_parquet(os.path.join(corpus_dir, f)) for f in query_files]
        self.queries = pd.concat(query_dfs, ignore_index=True)
        
        # 加载真实标签
        qrels_dir = parquet_dir + "-qrels"
        qrels_file = os.path.join(qrels_dir, "data", "dev-00000-of-00001-6c20deed08d304e4.parquet")
        self.qrels_data = pd.read_parquet(qrels_file)
        
        # 生成ID列表
        self.corpus_ids = list(range(len(self.corpus)))
        self.query_ids = list(range(len(self.queries)))


class MMarcoRetrievalCorpusDataset(Dataset):
    """
    MMarcoRetrieval语料库数据集类，用于加载和处理语料库文本数据
    """
    def __init__(self, mmarcoretrieval_dataset):
        """
        初始化MMarcoRetrieval语料库数据集
        
        Args:
            mmarcoretrieval_dataset: MMarcoRetrievalDataset实例
        """
        self.corpus = mmarcoretrieval_dataset.corpus
        self.corpus_ids = mmarcoretrieval_dataset.corpus_ids

    def __len__(self):
        """返回数据集大小"""
        return len(self.corpus_ids)

    def __getitem__(self, idx):
        """
        获取指定索引的语料文本数据
        
        Args:
            idx: 数据索引
            
        Returns:
            id: 语料ID
            text: 语料文本内容
        """
        return self.corpus.iloc[idx]["id"], self.corpus.iloc[idx]["text"]


class MMarcoRetrievalQueryDataset(Dataset):
    """
    MMarcoRetrieval查询数据集类，用于加载和处理查询文本数据
    """
    def __init__(self, mmarcoretrieval_dataset):
        """
        初始化MMarcoRetrieval查询数据集
        
        Args:
            mmarcoretrieval_dataset: MMarcoRetrievalDataset实例
        """
        self.queries = mmarcoretrieval_dataset.queries
        self.query_ids = mmarcoretrieval_dataset.query_ids

    def __len__(self):
        """返回数据集大小"""
        return len(self.query_ids)

    def __getitem__(self, idx):
        """
        获取指定索引的查询文本数据
        
        Args:
            idx: 数据索引
            
        Returns:
            id: 查询ID
            text: 查询文本内容
        """
        return self.queries.iloc[idx]["id"], self.queries.iloc[idx]["text"]


def mmarcoretrieval_get_corpus_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取MMarcoRetrieval语料库数据加载器
    
    Args:
        dataset: MMarcoRetrievalCorpusDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        语料库数据加载器
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def mmarcoretrieval_get_query_dataloader(dataset, batch_size=16, num_workers=1):
    """
    获取MMarcoRetrieval查询数据加载器
    
    Args:
        dataset: MMarcoRetrievalQueryDataset实例
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        
    Returns:
        查询数据加载器
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def mmarcoretrieval_generate_qrels(mmarcoretrieval_dataset):
    """
    生成查询相关性文件
    
    Args:
        mmarcoretrieval_dataset: MMarcoRetrievalDataset实例
        
    Returns:
        qrels: 查询相关性字典，格式为{查询ID: {文档ID: 相关性分数}}
    """
    qrels = {}
    for _, row in mmarcoretrieval_dataset.qrels_data.iterrows():
        query_id = row['qid']
        doc_id = row['pid']
        score = row['score']
        if query_id not in qrels:
            qrels[query_id] = {}
        if score == 1:
            qrels[query_id][doc_id] = 1
    return qrels


def mmarcoretrieval_generate_runs(model_wrapper, corpus_loader, query_loader, device):
    """
    生成检索结果和计算编码时间
    
    Args:
        model_wrapper: 模型包装器
        corpus_loader: 语料库数据加载器
        query_loader: 查询数据加载器
        device: 计算设备
        
    Returns:
        runs: 检索结果字典
        total_txt_time: 总文本编码时间
        total_txt_time: 总文本编码时间（重复参数）
        avg_txt_time: 平均文本编码时间
        avg_txt_time: 平均文本编码时间（重复参数）
    """
    runs = {}

    corpus_ids = []
    corpus_features = []
    corpus_times, query_times = [], []

    with torch.no_grad():
        # 语料库编码
        for ids, texts in tqdm(corpus_loader, desc="Extracting corpus features"):
            inputs = model_wrapper.preprocess_text(texts, device)
            start = time.time()
            corpus_feats = model_wrapper.get_text_features(inputs)
            if isinstance(corpus_feats, torch.Tensor):
                corpus_feats = corpus_feats.cpu().numpy()
            corpus_times.append(time.time() - start)

            corpus_ids.extend(ids)
            corpus_features.append(corpus_feats)

        corpus_features = np.vstack(corpus_features).astype('float32')
        faiss.normalize_L2(corpus_features)

        # 建立Faiss索引
        dim = corpus_features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_features)

    # 查询编码 + 检索
    query_features = []
    query_ids = []

    with torch.no_grad():
        for batch_ids, texts in tqdm(query_loader, desc="Extracting query features"):
            inputs = model_wrapper.preprocess_text(texts, device)
            start = time.time()
            query_feats = model_wrapper.get_text_features(inputs)
            if isinstance(query_feats, torch.Tensor):
                query_feats = query_feats.cpu().numpy()
            query_times.append(time.time() - start)

            query_ids.extend(batch_ids)
            query_features.append(query_feats)

    query_features = np.vstack(query_features).astype('float32')
    faiss.normalize_L2(query_features)

    # 查询检索
    sims, indices = index.search(query_features, k=10)

    # 生成检索结果
    for i, qid in enumerate(query_ids):
        runs[str(qid)] = {}
        for j in range(10):
            doc_id = corpus_ids[indices[i][j]]
            sim = sims[i][j]
            runs[str(qid)][str(doc_id)] = float(sim)

    total_corpus_time = sum(corpus_times)
    total_query_time = sum(query_times)
    
    # 计算平均编码时间
    num_corpus = len(corpus_loader.dataset)
    num_queries = len(query_loader.dataset)
    
    total_txt_time = total_corpus_time + total_query_time
    avg_txt_time = total_txt_time / (num_corpus + num_queries)

    return runs, total_txt_time, total_txt_time, avg_txt_time, avg_txt_time 