import torch
from torch.nn.functional import normalize
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, ClapModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers import BlipProcessor, BlipForImageTextRetrieval
from transformers import AlignProcessor, AlignModel
from transformers import AutoModel, AutoTokenizer
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from languagebind.audio.processing_audio import AudioTransform
from torchvision import transforms
import soundfile as sf
import io
import numpy as np
import torchaudio
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
from transformers import XCLIPProcessor, XCLIPModel
from VideoCLIP_XL.modeling import VideoCLIP_XL
from VideoCLIP_XL.utils.text_encoder import text_encoder

class CLIP4CLIPModelWrapper:
    def __init__(self, model_name="./download/models/clip4clip-webvid150k", mode="fp32"):
        """
        初始化视频和文本的 CLIP 模型。
        :param model_name: 模型路径或名称
        :param mode: 模型的精度模式，可以是 "fp32", "fp16", "int8", 或 "int4"
        """
        self.mode = mode

        # 加载视频模型
        if mode == "int4":
            self.video_model = CLIPVisionModelWithProjection.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            ).eval()
        elif mode == "int8":
            self.video_model = CLIPVisionModelWithProjection.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            ).eval()
        elif mode == "fp16":
            self.video_model = CLIPVisionModelWithProjection.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            ).eval()
        else:  # 默认 fp32
            self.video_model = CLIPVisionModelWithProjection.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            ).eval()

        # 加载文本模型
        if mode == "int4":
            self.text_model = CLIPTextModelWithProjection.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            ).eval()
        elif mode == "int8":
            self.text_model = CLIPTextModelWithProjection.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            ).eval()
        elif mode == "fp16":
            self.text_model = CLIPTextModelWithProjection.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            ).eval()
        else:  # 默认 fp32
            self.text_model = CLIPTextModelWithProjection.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            ).eval()

        # 加载 tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

    def get_video_features(self, video_frames, device='cuda:0'):
        """
        编码视频（从帧转换为视频特征）
        :param video_frames: 视频帧 tensor (batch_size, num_frames, C, H, W)
        :param device: 设备，默认 'cuda:0'
        :return: 视频特征 (batch_size, feature_dim)
        """
        B, F, C, H, W = video_frames.shape
        # 重塑为 [batch_size * num_frames, C, H, W]
        video_frames = video_frames.view(-1, C, H, W).to(device)

        with torch.no_grad():
            # 获取每一帧的特征
            video_features = self.video_model(video_frames)
            video_features = video_features["image_embeds"]
            # 重塑回 [batch_size, num_frames, feature_dim]
            video_features = video_features.view(B, F, -1)
            # 对每个视频的所有帧取平均得到视频特征
            video_features = torch.mean(video_features, dim=1)  # [batch_size, feature_dim]
            # 归一化
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_features

    def get_text_features(self, inputs, device='cuda:0'):
        """
        编码文本
        :param input_ids: 输入的文本 ID (batch_size, seq_len)
        :param device: 设备，默认 'cuda:0'
        """
        inputs = inputs.to(device)

        with torch.no_grad():
            text_features = self.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        # 规范化文本特征
        text_features = text_features[0]  # 取得最后的嵌入
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()

    def preprocess_text(self, texts, device='cuda:0'):
        """
        预处理文本数据
        :param texts: 输入文本
        :param device: 设备
        :return: 处理后的输入
        """
        return self.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP的默认最大长度
            return_tensors="pt"
        ).to(device)

    def preprocess_video(self, video_paths, frame_rate=1.0, size=224, num_frames=8):
        """
        支持单个视频路径或路径列表，返回 (batch, num_frames, 3, size, size) 或 (num_frames, 3, size, size)
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

        def process_one(video_path):
            cap = cv2.VideoCapture(video_path)
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps < 1:
                images = torch.zeros([num_frames, 3, size, size], dtype=torch.float32)
                print("ERROR: problem reading video file: ", video_path)
                return images
            frames_idx = np.linspace(0, frameCount - 1, num_frames, dtype=int)
            images = torch.zeros([num_frames, 3, size, size], dtype=torch.float32)
            for i, idx in enumerate(frames_idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images[i] = preprocess(size, Image.fromarray(frame).convert("RGB"))
            cap.release()
            return images

        if isinstance(video_paths, str):
            return process_one(video_paths)
        elif isinstance(video_paths, (list, tuple)):
            frames_list = [process_one(p) for p in video_paths]
            return torch.stack(frames_list)


class CLAPModelWrapper:
    def __init__(self, model_name="./download/models/larger_clap_general", mode="fp32"):
        """
        初始化 CLAP 模型和处理器
        :param model_name: 预训练模型路径或名称
        :param mode: 模型的精度模式，可以是 "fp32", "fp16", "int8", 或 "int4"
        """
        self.mode = mode

        # 根据精度模式加载模型
        if mode == "int4":
            self.model = ClapModel.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            )
        elif mode == "int8":
            self.model = ClapModel.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            )
        elif mode == "fp16":
            self.model = ClapModel.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            )
        else:  # 默认 fp32
            self.model = ClapModel.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            )

        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_audio_features(self, inputs, device='cuda:0'):
        """
        编码音频
        :param device: 设备，默认 'cuda:0'
        :return: numpy array of audio embeddings
        """
        inputs = inputs.to(device)

        with torch.no_grad():
            audio_features = self.model.get_audio_features(**inputs)

        return audio_features.cpu().numpy()

    def get_text_features(self, inputs, device='cuda:0'):
        """
        编码文本
        :param device: 设备，默认 'cuda:0'
        :return: numpy array of text embeddings
        """
        inputs = inputs.to(device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return text_features.cpu().numpy()

    def preprocess_audio(self, audios, sampling_rate=48000, device='cuda:0'):
        """
        预处理音频数据
        :param audios: 音频文件路径或字节数据列表
        :param sampling_rate: 采样率
        :param device: 设备
        :return: 处理后的音频输入
        """
        waveforms = []
        # 判断输入类型
        for audio in audios:
            if isinstance(audio, str):  # 路径
                waveform_np, sr = sf.read(audio)
            elif isinstance(audio, dict) and 'bytes' in audio:
                waveform_np, sr = sf.read(io.BytesIO(audio['bytes']))
            if len(waveform_np.shape) > 1:
                waveform_np = np.mean(waveform_np, axis=1)
            waveform = torch.from_numpy(waveform_np).float()
            if sr != sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, sampling_rate)
                waveform = resampler(waveform)
            waveforms.append(waveform)
        waveforms_np = [waveform.numpy() for waveform in waveforms]
        return self.processor(
            audios=waveforms_np,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        ).to(device)

    def preprocess_text(self, texts, device='cuda:0'):
        """
        预处理文本数据
        :param texts: 输入文本
        :param device: 设备
        :return: 处理后的输入
        """
        return self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        ).to(device)


class CLIPModelWrapper:
    def __init__(self, model_name="./download/models/clip-vit-base-patch32", mode="fp32"):
        """
        初始化 CLIP 模型和处理器
        :param model_name: 预训练模型路径或名称
        :param mode: 模型的精度模式，可以是 "fp32", "fp16", "int8", 或 "int4"
        """
        self.mode = mode

        # 根据精度模式加载模型
        if mode == "int4":
            self.model = CLIPModel.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            )
        elif mode == "int8":
            self.model = CLIPModel.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            )
        elif mode == "fp16":
            self.model = CLIPModel.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            )
        else:  # 默认 fp32
            self.model = CLIPModel.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            )

        self.processor = CLIPProcessor.from_pretrained(model_name)

    def preprocess_image(self, images, device='cuda:0'):
        """
        预处理图像数据
        :param images: 输入图像
        :param device: 设备
        :return: 处理后的输入
        """
        return self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            do_rescale=False
        ).to(device)

    def preprocess_text(self, texts, device='cuda:0'):
        """
        预处理文本数据
        :param texts: 输入文本
        :param device: 设备
        :return: 处理后的输入
        """
        return self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

    def get_image_features(self, images, device='cuda:0'):
        """
        编码图像
        :param images: 图像数据
        :param device: 设备，默认是 'cuda:0'
        """
        images = images.to(device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**images)
        return image_features

    def get_text_features(self, texts, device='cuda:0'):
        """
        编码文本
        :param text: 文本数据
        :param device: 设备，默认是 'cuda:0'
        """
        texts = texts.to(device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**texts)
        return text_features


class BLIPModelWrapper:
    def __init__(self, model_name="./download/models/blip-itm-base-coco", mode="fp32"):
        """
        初始化 BLIP 模型和处理器
        :param model_name: 预训练模型路径或名称
        :param mode: 模型的精度模式，可以是 "fp32", "fp16", "int8", 或 "int4"
        """
        self.mode = mode
        self.processor = BlipProcessor.from_pretrained(model_name)

        # 根据精度模式加载模型
        if mode == "int4":
            self.model = BlipForImageTextRetrieval.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            )
        elif mode == "int8":
            self.model = BlipForImageTextRetrieval.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            )
        elif mode == "fp16":
            self.model = BlipForImageTextRetrieval.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            )
        else:  # 默认 fp32
            self.model = BlipForImageTextRetrieval.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            )
        
        self.model.eval()

    def preprocess_image(self, images, device='cuda:0'):
        """
        预处理图像数据
        :param images: 输入图像
        :param device: 设备
        :return: 处理后的输入
        """
        return self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            do_rescale=False
        ).to(device)

    def preprocess_text(self, texts, device='cuda:0'):
        """
        预处理文本数据
        :param texts: 输入文本
        :param device: 设备
        :return: 处理后的输入
        """
        return self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

    def get_image_features(self, images, device='cuda:0'):
        """
        获取图像特征
        :param images: 图像数据
        :param device: 设备
        """
        images = images.to(device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**images)
            image_embeds = vision_outputs[0]
            image_features = normalize(self.model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_features

    def get_text_features(self, texts, device='cuda:0'):
        """
        获取文本特征
        :param texts: 文本数据
        :param device: 设备
        """
        texts = texts.to(device)
        with torch.no_grad():
            text_outputs = self.model.text_encoder(**texts)
            text_embeds = text_outputs[0]
            text_features = normalize(self.model.text_proj(text_embeds[:, 0, :]), dim=-1)
        return text_features


class ALIGNModelWrapper:
    def __init__(self, model_name="./download/models/align-base", mode="fp32"):
        """
        初始化 ALIGN 模型和处理器
        :param model_name: 预训练模型路径或名称
        :param mode: 模型的精度模式，可以是 "fp32", "fp16", "int8", 或 "int4"
        """
        self.mode = mode
        
        # 初始化处理器和模型
        self.processor = AlignProcessor.from_pretrained(model_name)

        # 根据精度模式加载模型
        if mode == "int4":
            self.model = AlignModel.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            ).eval()
        elif mode == "int8":
            self.model = AlignModel.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            ).eval()
        elif mode == "fp16":
            self.model = AlignModel.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            ).eval()
        else:  # 默认 fp32
            self.model = AlignModel.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            ).eval()

    def preprocess_image(self, images, device='cuda:0'):
        """
        预处理图像数据
        :param images: 输入图像
        :param device: 设备
        :return: 处理后的输入
        """
        return self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            do_rescale=False
        ).to(device)

    def preprocess_text(self, texts, device='cuda:0'):
        """
        预处理文本数据
        :param texts: 输入文本
        :param device: 设备
        :return: 处理后的输入
        """
        return self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

    def get_image_features(self, images, device='cuda:0'):
        """
        获取图像特征
        :param images: 图像数据
        :param device: 设备
        """
        images = images.to(device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**images)
        return image_features

    def get_text_features(self, texts, device='cuda:0'):
        """
        获取文本特征
        :param texts: 文本数据
        :param device: 设备
        """
        texts = texts.to(device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**texts)
        return text_features


class AudioTransformForBytes:
    def __init__(self, config):
        self.sample_rate = config.audio_sample_rate
        self.num_mel_bins = config.num_mel_bins
        self.target_length = config.target_length
        self.audio_mean = config.audio_mean
        self.audio_std = config.audio_std

    def __call__(self, waveform_and_sr):
        waveform, origin_sr = waveform_and_sr
        if self.sample_rate != origin_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=origin_sr, new_freq=self.sample_rate)
        # 补零到 window_size
        min_len = 400  # window_size
        if waveform.shape[1] < min_len:
            pad = min_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        mel = self.get_mel(waveform)
        if mel.shape[0] > self.target_length:
            chunk_frames = self.target_length
            total_frames = mel.shape[0]
            ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
            if len(ranges[1]) == 0:
                ranges[1] = [0]
            if len(ranges[2]) == 0:
                ranges[2] = [0]
            idx_front = np.random.choice(ranges[0])
            idx_middle = np.random.choice(ranges[1])
            idx_back = np.random.choice(ranges[2])
            mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
            mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
            mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
            mel_fusion = torch.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
        elif mel.shape[0] < self.target_length:
            n_repeat = int(self.target_length / mel.shape[0]) + 1
            mel = mel.repeat(n_repeat, 1)[:self.target_length, :]
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        else:
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        mel_fusion = mel_fusion.transpose(1, 2)
        mel_fusion = (mel_fusion - self.audio_mean) / (self.audio_std * 2)
        return mel_fusion

    def get_mel(self, audio_data):
        audio_data = audio_data - audio_data.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            audio_data,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=10,
        )
        return mel

class LanguageBindModelWrapper:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.clip_type = {
            'image': './download/models/LanguageBind_Image',
            'video': './download/models/LanguageBind_Video',
            'audio': './download/models/LanguageBind_Audio',
            # 'depth': './download/models/LanguageBind_Depth',
            # 'thermal': './download/models/LanguageBind_Thermal',
        }
        self.model = LanguageBind(clip_type=self.clip_type, cache_dir='./cache_dir').to(device)
        self.model.eval()
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
            self.clip_type['image'], cache_dir='./cache_dir/tokenizer_cache_dir')
        self.modality_transform = {c: transform_dict[c](self.model.modality_config[c]) for c in self.clip_type.keys()}
        self.audio_transform_bytes = AudioTransform(self.model.modality_config['audio'].vision_config)

    def preprocess_image(self, images, device='cuda:0'):
        if device is None:
            device = self.device
        
        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            mean = images.mean().item()
            if mean > 1 or mean < 0:
                normed = images
            else:
                normed = torch.clone(images)
                for i in range(3):
                    normed[:, i, :, :] = (normed[:, i, :, :] - OPENAI_DATASET_MEAN[i]) / OPENAI_DATASET_STD[i]
            return {"pixel_values": normed.to(device)}
        else:
            return to_device(self.modality_transform['image'](images), device)

    def preprocess_video(self, videos, device='cuda:0'):
        if device is None:
            device = self.device

        if isinstance(videos, tuple):
            videos = list(videos)

        return to_device(self.modality_transform['video'](videos), device)

    def preprocess_audio(self, audios, device='cuda:0'):
        if device is None:
            device = self.device
        
        if isinstance(audios, tuple):
            audios = list(audios)
        
        if isinstance(audios[0], str):
            inputs = to_device(self.modality_transform['audio'](audios), device)
            return inputs

        processed = []
        for audio in audios:
            if isinstance(audio, dict) and 'bytes' in audio:
                audio_bytes = audio['bytes']
            waveform, sr = sf.read(io.BytesIO(audio_bytes))
            waveform = torch.from_numpy(waveform).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)
            processed.append(self.audio_transform_bytes((waveform, sr)))
        processed = torch.stack(processed)
        return {'pixel_values': processed.to(device)}

    def preprocess_text(self, texts, device='cuda:0'):
        if device is None:
            device = self.device
        return to_device(self.tokenizer(texts, max_length=77, padding='max_length', truncation=True, return_tensors='pt'), device)

    def get_image_features(self, image_inputs, device='cuda:0'):
        with torch.no_grad():
            embeddings = self.model({'image': image_inputs})
        return embeddings['image'].cpu().numpy()

    def get_video_features(self, video_inputs, device='cuda:0'):
        with torch.no_grad():
            embeddings = self.model({'video': video_inputs})
        return embeddings['video'].cpu().numpy()

    def get_audio_features(self, audio_inputs, device='cuda:0'):
        with torch.no_grad():
            embeddings = self.model({'audio': audio_inputs})
        return embeddings['audio'].cpu().numpy()

    def get_text_features(self, text_inputs, device='cuda:0'):
        with torch.no_grad():
            embeddings = self.model({'language': text_inputs})
        return embeddings['language'].cpu().numpy()


class ImageBindModelWrapper:
    def __init__(self, model_path="./download/models/imagebind", mode="fp32"):
        """
        初始化 ImageBind 模型
        :param model_path: 模型路径或名称 
        :param mode: 模型的精度模式，可以是 "fp32", "fp16", "int8", 或 "int4"
        """
        self.mode = mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        
        # 根据精度模式设置模型
        if mode == "fp16":
            self.model = self.model.half()
        
        self.model = self.model.to(self.device)
        self.model.eval()

        # 定义图像预处理的均值和标准差
        self.OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        self.OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    
    def preprocess_image(self, images, device='cuda:0'):
        """
        预处理图像数据，支持两种输入格式：
        1. 预处理过的张量 (用于 flickr8k, flickr30k, coco 等数据集)
        2. 图像路径列表 (用于其他数据集)
        """
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            # 检查是否需要标准化
            mean = images.mean().item()
            if mean > 1 or mean < 0:  # 已经标准化过
                normed = images
            else:  # 需要标准化
                normed = torch.clone(images)
                for i in range(3):
                    normed[:, i, :, :] = (normed[:, i, :, :] - self.OPENAI_DATASET_MEAN[i]) / self.OPENAI_DATASET_STD[i]
            inputs = {ModalityType.VISION: normed.to(self.device)}
        else:  # 图像路径列表
            inputs = {
                ModalityType.VISION: data.load_and_transform_vision_data(images, self.device)
            }
        return inputs
    
    def preprocess_text(self, texts, device='cuda:0'):
        """
        预处理文本数据
        :param texts: 输入文本列表
        """
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(texts, self.device)
        }
        return inputs
    
    def preprocess_audio(self, audios, device='cuda:0'):
        """
        预处理音频数据，支持两种输入格式：
        1. audio_bytes (用于 AudioCaps 数据集)
        2. 音频路径列表 (用于 clotho 数据集)
        """
        if isinstance(audios, (list, tuple)) and isinstance(audios[0], dict) and 'bytes' in audios[0]:
            # AudioCaps 数据集：将音频字节转换为波形数据
            audio_inputs = []
            for audio in audios:
                waveform, sr = sf.read(io.BytesIO(audio['bytes']))
                waveform = torch.from_numpy(waveform).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.transpose(0, 1)
                audio_inputs.append({
                    'waveform': waveform,
                    'sample_rate': sr
                })
            inputs = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(
                    audio_inputs,
                    device=self.device
                )
            }
        else:
            # 音频路径列表
            inputs = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(
                    audios,
                    device=self.device
                )
            }
        return inputs

    def preprocess_video(self, videos, device='cuda:0'):
        """
        预处理视频数据，支持两种输入格式：
        1. 预处理过的张量
        2. 视频路径列表 (用于 msvd, msrvtt 数据集)
        """
        if isinstance(videos, torch.Tensor):
            inputs = {ModalityType.VISION: videos.to(self.device)}
        else:
            inputs = {
                ModalityType.VISION: data.load_and_transform_video_data(
                    videos,
                    device=self.device
                )
            }
        return inputs
    
    def get_embeddings(self, inputs):
        """
        获取特征嵌入
        :param inputs: 预处理后的输入
        :return: 特征嵌入
        """
        with torch.no_grad():
            embeddings = self.model(inputs)
        return {key: value.cpu().numpy() for key, value in embeddings.items()}
    

    def get_image_features(self, image_inputs, device='cuda:0'):
        """
        获取图像特征
        :param image_inputs: 预处理后的图像输入
        """
        embeddings = self.get_embeddings(image_inputs)
        return embeddings[ModalityType.VISION]
    
    def get_text_features(self, text_inputs, device='cuda:0'):
        """
        获取文本特征
        :param text_inputs: 预处理后的文本输入
        """
        embeddings = self.get_embeddings(text_inputs)
        return embeddings[ModalityType.TEXT]
    
    def get_audio_features(self, audio_inputs, device='cuda:0'):
        """
        获取音频特征
        :param audio_inputs: 预处理后的音频输入
        """
        embeddings = self.get_embeddings(audio_inputs)
        return embeddings[ModalityType.AUDIO]

    def get_video_features(self, video_inputs, device='cuda:0'):
        """
        获取视频特征
        :param video_inputs: 预处理后的视频输入
        """
        embeddings = self.get_embeddings(video_inputs)
        return embeddings[ModalityType.VISION]


class GTEModelWrapper:
    def __init__(self, model_name="./download/models/gte-base-zh", mode="fp32"):
        """
        初始化 GTE 模型和 tokenizer
        :param model_name: 预训练模型路径或名称
        :param mode: 模型的精度模式，可以是 "fp32", "fp16", "int8", 或 "int4"
        """
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 根据精度模式加载模型
        if mode == "int4":
            self.model = AutoModel.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            ).eval()
        elif mode == "int8":
            self.model = AutoModel.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            ).eval()
        elif mode == "fp16":
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            ).eval()
        else:  # 默认 fp32
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            ).eval()

    def preprocess_text(self, texts, device='cuda:0'):
        """
        预处理文本数据
        :param texts: 输入文本
        :param device: 设备
        :return: 处理后的输入
        """
        return self.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

    def get_text_features(self, inputs, device='cuda:0'):
        """
        获取文本特征
        :param inputs: 文本数据
        :param device: 设备
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings


class BGEModelWrapper:
    def __init__(self, model_name="./download/models/bge-large-zh-v1.5", mode="fp32"):
        """
        初始化 BGE 模型和 tokenizer
        :param model_name: 预训练模型路径或名称
        :param mode: 模型的精度模式，可以是 "fp32", "fp16", "int8", 或 "int4"
        """
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 根据精度模式加载模型
        if mode == "int4":
            self.model = AutoModel.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            ).eval()
        elif mode == "int8":
            self.model = AutoModel.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            ).eval()
        elif mode == "fp16":
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            ).eval()
        else:  # 默认 fp32
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            ).eval()

    def preprocess_text(self, texts, device='cuda:0'):
        """
        预处理文本数据
        :param texts: 输入文本
        :param device: 设备
        :return: 处理后的输入
        """
        # 为每个文本添加指令前缀
        return self.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

    def get_text_features(self, inputs, device='cuda:0'):
        """
        获取文本特征
        :param inputs: 文本数据
        :param device: 设备
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的表示
        return embeddings.cpu().numpy()


class M3EModelWrapper:
    def __init__(self, model_name="./download/models/m3e-base", mode="fp32"):
        """
        初始化 M3E 模型和 tokenizer
        :param model_name: 预训练模型路径或名称
        :param mode: 模型的精度模式，可以是 "fp32", "fp16", "int8", 或 "int4"
        """
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 根据精度模式加载模型
        if mode == "int4":
            self.model = AutoModel.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            ).eval()
        elif mode == "int8":
            self.model = AutoModel.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            ).eval()
        elif mode == "fp16":
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            ).eval()
        else:  # 默认 fp32
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            ).eval()

    def preprocess_text(self, texts, device='cuda:0'):
        """
        预处理文本数据
        :param texts: 输入文本
        :param device: 设备
        :return: 处理后的输入
        """
        return self.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

    def get_text_features(self, inputs, device='cuda:0'):
        """
        获取文本特征
        :param inputs: 文本数据
        :param device: 设备
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的表示
        return embeddings.cpu().numpy()


class XCLIPModelWrapper:
    def __init__(self, model_name="./download/models/xclip-base-patch32", mode="fp32"):
        self.mode = mode
        if mode == "int4":
            self.model = XCLIPModel.from_pretrained(
                model_name, load_in_4bit=True, device_map={"": 0}
            ).eval()
        elif mode == "int8":
            self.model = XCLIPModel.from_pretrained(
                model_name, load_in_8bit=True, device_map={"": 0}
            ).eval()
        elif mode == "fp16":
            self.model = XCLIPModel.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map={"": 0}
            ).eval()
        else:
            self.model = XCLIPModel.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map={"": 0}
            ).eval()
        self.processor = XCLIPProcessor.from_pretrained(model_name)

    def preprocess_video(self, video_paths, num_frames=8, device='cuda:0'):
        def extract_frames(video_path, num_frames):
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idxs = np.linspace(0, frame_count - 1, num_frames, dtype=int)
            frames = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            cap.release()
            return frames

        # 支持单个和批量
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        all_frames = [extract_frames(p, num_frames) for p in video_paths]
        return self.processor(videos=all_frames, return_tensors="pt").to(device)

    def preprocess_text(self, texts, device='cuda:0'):
        return self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

    def get_video_features(self, video_inputs, device='cuda:0'):
        with torch.no_grad():
            video_features = self.model.get_video_features(**video_inputs)
        return video_features.cpu().numpy()

    def get_text_features(self, text_inputs, device='cuda:0'):
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        return text_features.cpu().numpy()


class VideoCLIPXLModelWrapper:
    def __init__(self, model_path="./download/models/VideoCLIP-XL.bin", mode="fp32"):
        self.mode = mode
        self.model = VideoCLIP_XL()
        state_dict = torch.load(model_path, map_location="cuda:0")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(torch.device("cuda:0"))
        self.model.eval()

    def preprocess_video(self, video_paths, num_frames=8, device="cuda:0"):
        import cv2
        import numpy as np
        v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
        def normalize(data):
            return (data / 255.0 - v_mean) / v_std
        def _frame_from_video(video):
            while video.isOpened():
                success, frame = video.read()
                if success:
                    yield frame
                else:
                    break
        def video_preprocessing(video_path, fnum=num_frames):
            video = cv2.VideoCapture(video_path)
            frames = [x for x in _frame_from_video(video)]
            if len(frames) < fnum:
                frames += [frames[-1]] * (fnum - len(frames))
            step = max(1, len(frames) // fnum)
            frames = frames[::step][:fnum]
            vid_tube = []
            for fr in frames:
                fr = fr[:,:,::-1]
                fr = cv2.resize(fr, (224, 224))
                fr = np.expand_dims(normalize(fr), axis=(0, 1))
                vid_tube.append(fr)
            vid_tube = np.concatenate(vid_tube, axis=1)
            vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
            vid_tube = torch.from_numpy(vid_tube)
            return vid_tube
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        video_tensors = [video_preprocessing(p) for p in video_paths]
        video_inputs = torch.cat(video_tensors, 0).float().to(device)
        return video_inputs

    def preprocess_text(self, texts, device="cuda:0"):
        return text_encoder.tokenize(texts, truncate=True).to(device)

    def get_video_features(self, video_inputs, device="cuda:0"):
        with torch.no_grad():
            feats = self.model.vision_model.get_vid_features(video_inputs.to(device)).float()
        return feats.cpu().numpy()

    def get_text_features(self, text_inputs, device="cuda:0"):
        with torch.no_grad():
            feats = self.model.text_model.encode_text(text_inputs.to(device)).float()
        return feats.cpu().numpy()

