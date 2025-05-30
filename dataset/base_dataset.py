from dataset import flickr30k_dataset
from dataset import coco_dataset
from dataset import flickr8k_dataset
from dataset import audiocaps_dataset
from dataset import clotho_dataset
from dataset import msvd_dataset
from dataset import msrvtt_dataset
from dataset import duretrieval_dataset
from dataset import t2retrieval_dataset
from dataset import mmarcoretrieval_dataset


def get_dataset(dataset_name, type='full'):
    """
    根据 dataset_name 返回对应的 dataset 对象
    """
    if dataset_name == "flickr30k":
        # 加载Flickr30k数据集（图像-文本对）
        dataset_obj = flickr30k_dataset.Flickr30kDataset(
            image_dir="./download/datasets/flickr30k/flickr30k-images/flickr30k-images",
            csv_path="./download/datasets/flickr30k/flickr_annotations_30k.csv",
            type=type)

        # 创建图像和文本数据集
        image_dataset = flickr30k_dataset.Flickr30kImageDataset(dataset_obj)
        text_dataset = flickr30k_dataset.Flickr30kTextDataset(dataset_obj)
        # 创建数据加载器
        image_loader = flickr30k_dataset.flickr30k_get_image_dataloader(
            image_dataset)
        text_loader = flickr30k_dataset.flickr30k_get_text_dataloader(
            text_dataset)
        # 生成查询相关性评分和运行函数
        qrels = flickr30k_dataset.flickr30k_generate_qrels(dataset_obj)
        generate_runs = flickr30k_dataset.flickr30k_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "image_dataset": image_dataset,
            "text_dataset": text_dataset,
            "image_loader": image_loader,
            "text_loader": text_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "image_loader",  # 指定主要加载器（图像）
        }

    elif dataset_name == "coco":
        # 加载COCO数据集（图像-文本对）
        dataset_obj = coco_dataset.COCODataset(
            parquet_dir="./download/datasets/coco_captions/data",
            type=type)

        # 创建图像和文本数据集
        image_dataset = coco_dataset.COCOImageDataset(dataset_obj)
        text_dataset = coco_dataset.COCOTextDataset(dataset_obj)
        # 创建数据加载器
        image_loader = coco_dataset.coco_get_image_dataloader(image_dataset)
        text_loader = coco_dataset.coco_get_text_dataloader(text_dataset)
        # 生成查询相关性评分和运行函数
        qrels = coco_dataset.coco_generate_qrels(dataset_obj)
        generate_runs = coco_dataset.coco_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "image_dataset": image_dataset,
            "text_dataset": text_dataset,
            "image_loader": image_loader,
            "text_loader": text_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "image_loader",  # 指定主要加载器（图像）
        }

    elif dataset_name == "flickr8k":
        # 加载Flickr8k数据集（图像-文本对）
        dataset_obj = flickr8k_dataset.Flickr8kDataset(
            parquet_dir="./download/datasets/flickr8k/data",
            type=type)

        # 创建图像和文本数据集
        image_dataset = flickr8k_dataset.Flickr8kImageDataset(dataset_obj)
        text_dataset = flickr8k_dataset.Flickr8kTextDataset(dataset_obj)
        # 创建数据加载器
        image_loader = flickr8k_dataset.flickr8k_get_image_dataloader(
            image_dataset)
        text_loader = flickr8k_dataset.flickr8k_get_text_dataloader(
            text_dataset)
        # 生成查询相关性评分和运行函数
        qrels = flickr8k_dataset.flickr8k_generate_qrels(dataset_obj)
        generate_runs = flickr8k_dataset.flickr8k_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "image_dataset": image_dataset,
            "text_dataset": text_dataset,
            "image_loader": image_loader,
            "text_loader": text_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "image_loader",  # 指定主要加载器（图像）
        }

    elif dataset_name == "audiocaps":
        # 加载AudioCaps数据集（音频-文本对）
        dataset_obj = audiocaps_dataset.AudioCapsDataset(
            parquet_dir="./download/datasets/AudioCaps/data",
            type=type)

        # 创建音频和文本数据集
        audio_dataset = audiocaps_dataset.AudioCapsAudioDataset(dataset_obj)
        text_dataset = audiocaps_dataset.AudioCapsTextDataset(dataset_obj)
        # 创建数据加载器
        audio_loader = audiocaps_dataset.audiocaps_get_audio_dataloader(
            audio_dataset)
        text_loader = audiocaps_dataset.audiocaps_get_text_dataloader(
            text_dataset)
        # 生成查询相关性评分和运行函数
        qrels = audiocaps_dataset.audiocaps_generate_qrels(dataset_obj)
        generate_runs = audiocaps_dataset.audiocaps_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "audio_dataset": audio_dataset,
            "text_dataset": text_dataset,
            "audio_loader": audio_loader,
            "text_loader": text_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "audio_loader",  # 指定主要加载器（音频）
        }

    elif dataset_name == "clotho":
        # 加载Clotho数据集（音频-文本对）
        dataset_obj = clotho_dataset.ClothoDataset(
            root_dir="./download/datasets/Clotho",
            type=type)

        # 创建音频和文本数据集
        audio_dataset = clotho_dataset.ClothoAudioDataset(dataset_obj)
        text_dataset = clotho_dataset.ClothoTextDataset(dataset_obj)
        # 创建数据加载器
        audio_loader = clotho_dataset.clotho_get_audio_dataloader(
            audio_dataset)
        text_loader = clotho_dataset.clotho_get_text_dataloader(text_dataset)
        # 生成查询相关性评分和运行函数
        qrels = clotho_dataset.clotho_generate_qrels(dataset_obj)
        generate_runs = clotho_dataset.clotho_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "audio_dataset": audio_dataset,
            "text_dataset": text_dataset,
            "audio_loader": audio_loader,
            "text_loader": text_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "audio_loader",  # 指定主要加载器（音频）
        }

    elif dataset_name == "msvd":
        # 加载MSVD数据集（视频-文本对）
        dataset_obj = msvd_dataset.MSVDDataset(
            root_dir="./download/datasets/MSVD"
        )

        # 创建视频和文本数据集
        video_dataset = msvd_dataset.MSVDVideoDataset(dataset_obj)
        text_dataset = msvd_dataset.MSVDTextDataset(dataset_obj)
        # 创建数据加载器
        video_loader = msvd_dataset.msvd_get_video_dataloader(video_dataset)
        text_loader = msvd_dataset.msvd_get_text_dataloader(text_dataset)
        # 生成查询相关性评分和运行函数
        qrels = msvd_dataset.msvd_generate_qrels(dataset_obj)
        generate_runs = msvd_dataset.msvd_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "video_dataset": video_dataset,
            "text_dataset": text_dataset,
            "video_loader": video_loader,
            "text_loader": text_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "video_loader",  # 指定主要加载器（视频）
        }

    elif dataset_name == "msrvtt":
        # 加载MSR-VTT数据集（视频-文本对）
        dataset_obj = msrvtt_dataset.MSRVTTDataset(
            root_dir="./download/datasets/MSR-VTT",
            type=type
        )

        # 创建视频和文本数据集
        video_dataset = msrvtt_dataset.MSRVTTVideoDataset(dataset_obj)
        text_dataset = msrvtt_dataset.MSRVTTTextDataset(dataset_obj)
        # 创建数据加载器
        video_loader = msrvtt_dataset.msrvtt_get_video_dataloader(video_dataset)
        text_loader = msrvtt_dataset.msrvtt_get_text_dataloader(text_dataset)
        # 生成查询相关性评分和运行函数
        qrels = msrvtt_dataset.msrvtt_generate_qrels(dataset_obj)
        generate_runs = msrvtt_dataset.msrvtt_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "video_dataset": video_dataset,
            "text_dataset": text_dataset,
            "video_loader": video_loader,
            "text_loader": text_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "video_loader",  # 指定主要加载器（视频）
        }

    elif dataset_name == "duretrieval":
        # 加载DuRetrieval数据集（文本-文本检索）
        dataset_obj = duretrieval_dataset.DuRetrievalDataset(
            parquet_dir="./download/datasets/DuRetrieval",
            type=type
        )

        # 创建语料库和查询数据集
        corpus_dataset = duretrieval_dataset.DuRetrievalCorpusDataset(dataset_obj)
        query_dataset = duretrieval_dataset.DuRetrievalQueryDataset(dataset_obj)
        # 创建数据加载器
        corpus_loader = duretrieval_dataset.duretrieval_get_corpus_dataloader(corpus_dataset)
        query_loader = duretrieval_dataset.duretrieval_get_query_dataloader(query_dataset)
        # 生成查询相关性评分和运行函数
        qrels = duretrieval_dataset.duretrieval_generate_qrels(dataset_obj)
        generate_runs = duretrieval_dataset.duretrieval_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "corpus_dataset": corpus_dataset,
            "query_dataset": query_dataset,
            "corpus_loader": corpus_loader,
            "query_loader": query_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "corpus_loader",  # 指定主要加载器（语料库）
        }

    elif dataset_name == "t2retrieval":
        # 加载T2Retrieval数据集（文本-文本检索）
        dataset_obj = t2retrieval_dataset.T2RetrievalDataset(
            parquet_dir="./download/datasets/T2Retrieval",
            type=type
        )

        # 创建语料库和查询数据集
        corpus_dataset = t2retrieval_dataset.T2RetrievalCorpusDataset(dataset_obj)
        query_dataset = t2retrieval_dataset.T2RetrievalQueryDataset(dataset_obj)
        # 创建数据加载器
        corpus_loader = t2retrieval_dataset.t2retrieval_get_corpus_dataloader(corpus_dataset)
        query_loader = t2retrieval_dataset.t2retrieval_get_query_dataloader(query_dataset)
        # 生成查询相关性评分和运行函数
        qrels = t2retrieval_dataset.t2retrieval_generate_qrels(dataset_obj)
        generate_runs = t2retrieval_dataset.t2retrieval_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "corpus_dataset": corpus_dataset,
            "query_dataset": query_dataset,
            "corpus_loader": corpus_loader,
            "query_loader": query_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "corpus_loader",  # 指定主要加载器（语料库）
        }

    elif dataset_name == "mmarcoretrieval":
        # 加载MMarcoRetrieval数据集（多语言文本检索）
        dataset_obj = mmarcoretrieval_dataset.MMarcoRetrievalDataset(
            parquet_dir="./download/datasets/MMarcoRetrieval",
            type=type
        )

        # 创建语料库和查询数据集
        corpus_dataset = mmarcoretrieval_dataset.MMarcoRetrievalCorpusDataset(dataset_obj)
        query_dataset = mmarcoretrieval_dataset.MMarcoRetrievalQueryDataset(dataset_obj)
        # 创建数据加载器
        corpus_loader = mmarcoretrieval_dataset.mmarcoretrieval_get_corpus_dataloader(corpus_dataset)
        query_loader = mmarcoretrieval_dataset.mmarcoretrieval_get_query_dataloader(query_dataset)
        # 生成查询相关性评分和运行函数
        qrels = mmarcoretrieval_dataset.mmarcoretrieval_generate_qrels(dataset_obj)
        generate_runs = mmarcoretrieval_dataset.mmarcoretrieval_generate_runs

        # 返回包含所有数据集组件的字典
        return {
            "dataset_obj": dataset_obj,
            "corpus_dataset": corpus_dataset,
            "query_dataset": query_dataset,
            "corpus_loader": corpus_loader,
            "query_loader": query_loader,
            "qrels": qrels,
            "generate_runs": generate_runs,
            "primary_loader_key": "corpus_loader",  # 指定主要加载器（语料库）
        }

    else:
        # 不支持的数据集
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return None
