import argparse
import os
import sys
import logging
import time
import numpy as np
import speech_recognition as sr

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设备检测与选择
def detect_device(prefer_cuda=False, prefer_mlx=False):
    if prefer_mlx:
        try:
            import mlx.core as mx
            logger.info("Apple Silicon GPU (MLX) 支持已启用")
            return "mlx"
        except ImportError:
            logger.warning("未检测到MLX，将尝试CUDA/CPU")
    if prefer_cuda:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA可用，使用GPU")
            return "cuda"
        else:
            logger.warning("CUDA不可用，使用CPU")
    logger.info("使用CPU")
    return "cpu"

# CLI参数
def parse_args():
    parser = argparse.ArgumentParser(description="多平台语音识别+TTS+LLM角色扮演 Demo")
    parser.add_argument("--enable-llm", action="store_true", help="启用LLM润色输出")
    parser.add_argument("--tts-url", type=str, default="http://127.0.0.1:9880/", help="TTS服务地址")
    parser.add_argument("--llm-url", type=str, default="http://127.0.0.1:1234/v1", help="LLM服务地址")
    parser.add_argument("--llm-key", type=str, default=None, help="LLM API Key")
    parser.add_argument("--llm-model", type=str, default="lmstudio-community/qwen2.5-3b-gguf/qwen-2.5-3b-instruct-10e-gguf_q8_0.gguf", help="LLM模型名")
    parser.add_argument("--device", choices=["cpu", "cuda", "mlx"], default="auto", help="推理设备：cpu/cuda/mlx/auto")
    return parser.parse_args()

def main():
    args = parse_args()
    # 设备选择
    if args.device == "auto":
        DEVICE = detect_device(prefer_cuda=True, prefer_mlx=True)
    else:
        DEVICE = args.device

    # 加载ASR模型
    from funasr import AutoModel
    print("正在加载FunASR模型...")
    funasr_model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_revision="v2.0.4",
        punc_model="ct-punc-c",
        punc_revision="v2.0.4",
        device=DEVICE
    )
    print("ASR模型加载完成。")

    # 可选加载说话人识别
    try:
        from modelscope.pipelines import pipeline
        sv_pipeline = pipeline(
            task='speaker-verification',
            model='iic/speech_campplus_sv_zh-cn_16k-common',
            model_revision='v1.0.0',
            device=DEVICE if DEVICE != "mlx" else "cpu"  # speaker verification一般无mlx
        )
        enroll_audio_dir = "./speaker_audio"
        enroll_audio_paths = []
        if os.path.exists(enroll_audio_dir) and os.path.isdir(enroll_audio_dir):
            enroll_audio_paths = [os.path.join(enroll_audio_dir, f) for f in os.listdir(enroll_audio_dir) 
                                if f.endswith(('.wav', '.mp3', '.flac'))]
    except Exception as e:
        sv_pipeline = None
        enroll_audio_paths = []
        logger.warning(f"说话人识别加载失败: {str(e)}")

    def recognize_funasr(audio_data):
        audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0
        result = funasr_model.generate(
            input=[audio_np],
            batch_size_s=50,
            hotword='',
        )
        if result and isinstance(result, list) and len(result) > 0:
            return result[0].get("text", "")
        return ""
    
    def verify_speaker(audio_data):
        if not enroll_audio_paths or not sv_pipeline:
            return [True, None]
        enroll_audio = enroll_audio_paths[0]
        try:
            audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / 32768.0
            result = sv_pipeline([enroll_audio, audio_np],output_emb=True,thr=0.20)
            if result['outputs'] and "text" in result['outputs'] and result['outputs']["text"] == "yes":
                return [True,result['embs']]
            return [False,result['embs']]
        except Exception as e:
            logger.error(f"说话人验证失败: {str(e)}")
            return [False,None]
    
    # 初始化TTS客户端
    import client
    tts_client = client.TTSStreamingClient(
        api_url=args.tts_url,
        sample_rate=16000,
        channels=1,
        chunk_size=128
    )
    # 初始化LLM
    if args.enable_llm:
        import firefly_rewriter
        llm_rewriter = firefly_rewriter.LiuYingCosplayRewriter(
            api_key=args.llm_key,
            model=args.llm_model,
            base_url=args.llm_url
        )
        logger.info("LLM润色已启用")
    else:
        llm_rewriter = None
        logger.info("未启用LLM润色")

    # 语音识别流程
    r = sr.Recognizer()
    r.non_speaking_duration = 0.1
    r.pause_threshold = 0.4
    mic = sr.Microphone(sample_rate=16000)
    with mic as source:
        print("采集环境噪声...")
        r.adjust_for_ambient_noise(source, duration=1.0)
        print(f"环境噪声阈值: {r.energy_threshold:.2f}")

    print("按 Ctrl+C 退出。参数适用于短句识别")
    print(f"说话人验证: {'启用' if enroll_audio_paths else '关闭'}，当前设备: {DEVICE}")

    while True:
        try:
            with mic as source:
                print("\n请说话（短句）...")
                audio = r.listen(source)
                print("语音结束，处理中...")
            # 说话人验证
            if enroll_audio_paths:
                is_target_speaker = verify_speaker(audio)
                if not is_target_speaker[0]:
                    print("非目标说话人，跳过。")
                    continue
            # ASR识别
            result = recognize_funasr(audio)
            print("识别文本：", result)
            # LLM润色（可选）
            if llm_rewriter:
                result = llm_rewriter.rewrite(result)
                print("LLM润色后文本：", result)
            # TTS合成
            tts_client.stream_tts(result[:30], text_language="zh")
        except KeyboardInterrupt:
            print("\n程序已退出")
            break
        except Exception as e:
            logger.error(f"主循环异常: {str(e)}")

if __name__ == "__main__":
    main()
