#!/bin/bash

# 设置结果保存目录
RESULTS_DIR="./results"
mkdir -p $RESULTS_DIR

# 设置模式列表
MODES="fp32"

# 文本-图像模型和数据集
run_image_text_models() {
    echo "===== 开始测试文本-图像模型 ====="
    
    # CLIP模型
    echo "测试CLIP模型..."
    python main.py --model_type clip --model_path ./download/models/clip-vit-base-patch32 --modes $MODES --save_path $RESULTS_DIR/clip_results.csv --dataset_name flickr30k --dataset_type test
    python main.py --model_type clip --model_path ./download/models/clip-vit-base-patch32 --modes $MODES --save_path $RESULTS_DIR/clip_results.csv --dataset_name flickr8k --dataset_type test
    python main.py --model_type clip --model_path ./download/models/clip-vit-base-patch32 --modes $MODES --save_path $RESULTS_DIR/clip_results.csv --dataset_name coco --dataset_type test
    
    # BLIP模型
    echo "测试BLIP模型..."
    python main.py --model_type blip --model_path ./download/models/blip-itm-base-coco --modes $MODES --save_path $RESULTS_DIR/blip_results.csv --dataset_name flickr30k --dataset_type test
    python main.py --model_type blip --model_path ./download/models/blip-itm-base-coco --modes $MODES --save_path $RESULTS_DIR/blip_results.csv --dataset_name flickr8k --dataset_type test
    python main.py --model_type blip --model_path ./download/models/blip-itm-base-coco --modes $MODES --save_path $RESULTS_DIR/blip_results.csv --dataset_name coco --dataset_type test
    
    # ALIGN模型
    echo "测试ALIGN模型..."
    python main.py --model_type align --model_path ./download/models/align-base --modes $MODES --save_path $RESULTS_DIR/align_results.csv --dataset_name flickr30k --dataset_type test
    python main.py --model_type align --model_path ./download/models/align-base --modes $MODES --save_path $RESULTS_DIR/align_results.csv --dataset_name flickr8k --dataset_type test
    python main.py --model_type align --model_path ./download/models/align-base --modes $MODES --save_path $RESULTS_DIR/align_results.csv --dataset_name coco --dataset_type test
    
    # ImageBind模型（图像部分）
    echo "测试ImageBind模型（图像部分）..."
    python main.py --model_type imagebind --model_path ./download/models/imagebind --modes $MODES --save_path $RESULTS_DIR/imagebind_results.csv --dataset_name flickr30k --dataset_type test
    python main.py --model_type imagebind --model_path ./download/models/imagebind --modes $MODES --save_path $RESULTS_DIR/imagebind_results.csv --dataset_name flickr8k --dataset_type test
    python main.py --model_type imagebind --model_path ./download/models/imagebind --modes $MODES --save_path $RESULTS_DIR/imagebind_results.csv --dataset_name coco --dataset_type test
    
    # LanguageBind模型（图像部分）
    echo "测试LanguageBind模型（图像部分）..."
    python main.py --model_type languagebind --model_path ./download/models/languagebind --modes $MODES --save_path $RESULTS_DIR/languagebind_results.csv --dataset_name flickr30k --dataset_type test
    python main.py --model_type languagebind --model_path ./download/models/languagebind --modes $MODES --save_path $RESULTS_DIR/languagebind_results.csv --dataset_name flickr8k --dataset_type test
    python main.py --model_type languagebind --model_path ./download/models/languagebind --modes $MODES --save_path $RESULTS_DIR/languagebind_results.csv --dataset_name coco --dataset_type test
    
    echo "===== 文本-图像模型测试完成 ====="
}

# 文本-音频模型和数据集
run_audio_text_models() {
    echo "===== 开始测试文本-音频模型 ====="
    
    # CLAP模型
    echo "测试CLAP模型..."
    python main.py --model_type clap --model_path ./download/models/larger_clap_general --modes $MODES --save_path $RESULTS_DIR/clap_results.csv --dataset_name audiocaps --dataset_type test
    python main.py --model_type clap --model_path ./download/models/larger_clap_general --modes $MODES --save_path $RESULTS_DIR/clap_results.csv --dataset_name clotho --dataset_type test
    
    # ImageBind模型（音频部分）
    echo "测试ImageBind模型（音频部分）..."
    python main.py --model_type imagebind --model_path ./download/models/imagebind --modes $MODES --save_path $RESULTS_DIR/imagebind_results.csv --dataset_name audiocaps --dataset_type test
    python main.py --model_type imagebind --model_path ./download/models/imagebind --modes $MODES --save_path $RESULTS_DIR/imagebind_results.csv --dataset_name clotho --dataset_type test
    
    # LanguageBind模型（音频部分）
    echo "测试LanguageBind模型（音频部分）..."
    python main.py --model_type languagebind --model_path ./download/models/languagebind --modes $MODES --save_path $RESULTS_DIR/languagebind_results.csv --dataset_name audiocaps --dataset_type test
    python main.py --model_type languagebind --model_path ./download/models/languagebind --modes $MODES --save_path $RESULTS_DIR/languagebind_results.csv --dataset_name clotho --dataset_type test
    
    echo "===== 文本-音频模型测试完成 ====="
}

# 文本-视频模型和数据集
run_video_text_models() {
    echo "===== 开始测试文本-视频模型 ====="
    
    # CLIP4CLIP模型
    echo "测试CLIP4CLIP模型..."
    python main.py --model_type clip4clip --model_path ./download/models/clip4clip-webvid150k --modes $MODES --save_path $RESULTS_DIR/clip4clip_results.csv --dataset_name msvd --dataset_type test
    python main.py --model_type clip4clip --model_path ./download/models/clip4clip-webvid150k --modes $MODES --save_path $RESULTS_DIR/clip4clip_results.csv --dataset_name msrvtt --dataset_type test

    # XCLIP模型
    echo "测试XCLIP模型..."
    python main.py --model_type xclip --model_path ./download/models/xclip-base-patch32 --modes $MODES --save_path $RESULTS_DIR/xclip_results.csv --dataset_name msvd --dataset_type test
    python main.py --model_type xclip --model_path ./download/models/xclip-base-patch32 --modes $MODES --save_path $RESULTS_DIR/xclip_results.csv --dataset_name msrvtt --dataset_type test

    # VideoCLIP-XL模型
    echo "测试VideoCLIP-XL模型..."
    python main.py --model_type videoclipxl --model_path ./download/models/VideoCLIP-XL.bin --modes $MODES --save_path $RESULTS_DIR/videoclipxl_results.csv --dataset_name msvd --dataset_type test
    python main.py --model_type videoclipxl --model_path ./download/models/VideoCLIP-XL.bin --modes $MODES --save_path $RESULTS_DIR/videoclipxl_results.csv --dataset_name msrvtt --dataset_type test

    # ImageBind模型（视频部分）
    echo "测试ImageBind模型（视频部分）..."
    python main.py --model_type imagebind --model_path ./download/models/imagebind --modes $MODES --save_path $RESULTS_DIR/imagebind_results.csv --dataset_name msvd --dataset_type test
    python main.py --model_type imagebind --model_path ./download/models/imagebind --modes $MODES --save_path $RESULTS_DIR/imagebind_results.csv --dataset_name msrvtt --dataset_type test
    
    # LanguageBind模型（视频部分）
    echo "测试LanguageBind模型（视频部分）..."
    python main.py --model_type languagebind --model_path ./download/models/languagebind --modes $MODES --save_path $RESULTS_DIR/languagebind_results.csv --dataset_name msvd --dataset_type test
    python main.py --model_type languagebind --model_path ./download/models/languagebind --modes $MODES --save_path $RESULTS_DIR/languagebind_results.csv --dataset_name msrvtt --dataset_type test
    
    echo "===== 文本-视频模型测试完成 ====="
}

# 纯文本模型和数据集
run_text_models() {
    echo "===== 开始测试纯文本模型 ====="
    
    # GTE模型
    echo "测试GTE模型..."
    python main.py --model_type gte --model_path ./download/models/gte-base-zh --modes $MODES --save_path $RESULTS_DIR/gte_results.csv --dataset_name t2retrieval --dataset_type test
    python main.py --model_type gte --model_path ./download/models/gte-base-zh --modes $MODES --save_path $RESULTS_DIR/gte_results.csv --dataset_name mmarcoretrieval --dataset_type test
    python main.py --model_type gte --model_path ./download/models/gte-base-zh --modes $MODES --save_path $RESULTS_DIR/gte_results.csv --dataset_name duretrieval --dataset_type test
    
    # BGE模型
    echo "测试BGE模型..."
    python main.py --model_type bge --model_path ./download/models/bge-base-zh-v1.5 --modes $MODES --save_path $RESULTS_DIR/bge_results.csv --dataset_name t2retrieval --dataset_type test
    python main.py --model_type bge --model_path ./download/models/bge-base-zh-v1.5 --modes $MODES --save_path $RESULTS_DIR/bge_results.csv --dataset_name mmarcoretrieval --dataset_type test
    python main.py --model_type bge --model_path ./download/models/bge-base-zh-v1.5 --modes $MODES --save_path $RESULTS_DIR/bge_results.csv --dataset_name duretrieval --dataset_type test
    
    # M3E模型
    echo "测试M3E模型..."
    python main.py --model_type m3e --model_path ./download/models/m3e-base --modes $MODES --save_path $RESULTS_DIR/m3e_results.csv --dataset_name t2retrieval --dataset_type test
    python main.py --model_type m3e --model_path ./download/models/m3e-base --modes $MODES --save_path $RESULTS_DIR/m3e_results.csv --dataset_name mmarcoretrieval --dataset_type test
    python main.py --model_type m3e --model_path ./download/models/m3e-base --modes $MODES --save_path $RESULTS_DIR/m3e_results.csv --dataset_name duretrieval --dataset_type test
    
    echo "===== 纯文本模型测试完成 ====="
}

# 运行所有测试
run_all_tests() {
    echo "===== 开始运行所有模型测试 ====="
    run_image_text_models
    run_audio_text_models
    run_video_text_models
    run_text_models
    echo "===== 所有模型测试完成 ====="
}

# 根据命令行参数运行特定类型的测试
if [ $# -eq 0 ]; then
    # 如果没有参数，运行所有测试
    run_all_tests
else
    # 根据参数运行特定测试
    case "$1" in
        "image")
            run_image_text_models
            ;;
        "audio")
            run_audio_text_models
            ;;
        "video")
            run_video_text_models
            ;;
        "text")
            run_text_models
            ;;
        *)
            echo "未知参数: $1"
            echo "可用参数: image, audio, video, text"
            exit 1
            ;;
    esac
fi

echo "测试完成！" 