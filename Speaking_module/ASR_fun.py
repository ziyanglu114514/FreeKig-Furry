import speech_recognition as sr
from funasr import AutoModel
import client
import firefly_rewriter
import time
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import logging
import tempfile

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 检查是否支持MLX
try:
    import mlx.core as mx
    DEVICE = "mlx"
    logger.info("Apple Silicon GPU (MLX) 支持已启用")
except ImportError:
    DEVICE = "cpu"
    logger.warning("未找到MLX支持，将使用CPU")

# 1. 加载FunASR模型
print("正在加载FunASR模型...")
funasr_model = AutoModel(
    model="paraformer-zh",
    model_revision="v2.0.4",
    vad_model="fsmn-vad",
    vad_revision="v2.0.4",
    punc_model="ct-punc-c",
    punc_revision="v2.0.4",
    device=DEVICE  # 使用MLX或CPU
)
print("模型加载完成。")

# 2. 加载说话人识别模型
print("正在加载说话人识别模型...")
try:
    sv_pipeline = pipeline(
        task='speaker-verification',
        model='iic/speech_campplus_sv_zh-cn_16k-common',
        model_revision='v1.0.0',
        device="gpu"  # 使用MLX或CPU
    )
    print("说话人识别模型加载完成。")
    
    # 3. 注册目标说话人
    enroll_audio_dir = "./speaker_audio"
    enroll_audio_paths = []
    threshold = 0.5
    
    if os.path.exists(enroll_audio_dir) and os.path.isdir(enroll_audio_dir):
        enroll_audio_paths = [os.path.join(enroll_audio_dir, f) for f in os.listdir(enroll_audio_dir) 
                              if f.endswith(('.wav', '.mp3', '.flac'))]
        
        if enroll_audio_paths:
            logger.info(f"找到 {len(enroll_audio_paths)} 个注册音频文件")
        else:
            logger.warning("注册音频目录为空")
    else:
        logger.warning(f"未找到注册音频目录: {enroll_audio_dir}")
except Exception as e:
    logger.error(f"加载说话人识别模型失败: {str(e)}")
    sv_pipeline = None
    enroll_audio_paths = []

def recognize_funasr(audio_data):
    """使用FunASR进行非流式语音识别"""
    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
    audio_np = audio_np.astype(np.float32) / 32768.0
    
    # 关键优化：降低批处理大小
    result = funasr_model.generate(
        input=[audio_np],
        batch_size_s=50,  # 优化为短句处理
        hotword='',
    )
    
    if result and isinstance(result, list) and len(result) > 0:
        return result[0].get("text", "")
    return ""

def verify_speaker(audio_data):
    """验证说话人是否匹配注册的说话人"""
    if not enroll_audio_paths or not sv_pipeline:
        return True
    
    # 优化：只使用第一个注册音频验证
    enroll_audio = enroll_audio_paths[0]
    
    try:
        # 将音频数据转换为numpy数组
        audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0
        
        # 直接使用内存中的音频数据进行验证
        # 注意：这里假设sv_pipeline可以直接处理numpy数组
        # 如果模型需要文件路径，则需要修改为内存处理方式
        result = sv_pipeline([enroll_audio, audio_np],output_emb=True)
        
        if result['outputs'] and "text" in result['outputs'] and result['outputs']["text"] == "yes":
            logger.info(f"说话人验证通过: {os.path.basename(enroll_audio)}")
            return [True,result['embs']]
        elif result['outputs']:
            score = result['outputs'].get("score", 0)
            logger.info(f"说话人不匹配: {os.path.basename(enroll_audio)}, 得分: {score:.4f}")
        return [False,result['embs']]
    except Exception as e:
        logger.error(f"说话人验证失败: {str(e)}")
        return [False,result['embs']]

# 2. 初始化麦克风和识别器
r = sr.Recognizer()
# 关键优化：VAD参数调整
r.non_speaking_duration = 0.1   # 降低非说话时间
r.pause_threshold = 0.4         # 大幅降低停顿阈值（适合短句）

mic = sr.Microphone(sample_rate=16000)

# 初始化TTS客户端
client = client.TTSStreamingClient(
        api_url="http://127.0.0.1:9880/",
        sample_rate=24000,
        channels=1,
        chunk_size=256
    )

client.stream_tts("测试输出", text_language="zh")
# 初始化LLM
liuying = firefly_rewriter.LiuYingCosplayRewriter(
        api_key=None,
        model="lmstudio-community/qwen2.5-3b-gguf/qwen-2.5-3b-instruct-10e-gguf_q8_0.gguf",
        base_url="http://127.0.0.1:1234/v1"
    )

with mic as source:
    # 优化：减少噪声采样时间
    print("正在采集环境噪声...")
    r.adjust_for_ambient_noise(source, duration=1.0)  # 减少采样时间
    print(f"环境噪声能量阈值已设置为 {r.energy_threshold:.2f}")

print("按 Ctrl+C 停止。优化参数适用于短句识别")
print(f"说话人验证状态: {'已启用' if enroll_audio_paths else '未启用'}")
print(f"当前设备: {'Apple Silicon GPU (MLX)' if DEVICE == 'mlx' else 'CPU'}")

while True:
    try:
        with mic as source:
            print("\n请说短句子(20字左右)...")
            audio = r.listen(source)
            speech_end_time = time.time()
            print("检测到语音结束")

        # 说话人验证（优化为单音频验证）
        if enroll_audio_paths:
            print("快速说话人验证中...")
            is_target_speaker = verify_speaker(audio)
            if not is_target_speaker[0]:
                recog_end_time = time.time()
                print(f"说话人识别处理耗时：{recog_end_time - speech_end_time:.3f}秒")
                print("非目标说话人，跳过")
                continue
        recog_end_time = time.time()
        print(f"说话人识别处理耗时：{recog_end_time - speech_end_time:.3f}秒")
        print("识别中...")
        speech_end_time = time.time()
        result = recognize_funasr(audio)
        print("识别结果：", result)
        
        recog_end_time = time.time()
        print(f"ASR处理耗时：{recog_end_time - speech_end_time:.3f}秒")

        # 可选LLM处理
        # result = liuying.rewrite(result)
        # print("LLM_output", result)

        tts_start_time = time.time()
        client.stream_tts(result[:30], text_language="zh")  # 限制输出长度
        tts_end_time = time.time()
        print(f"TTS处理耗时：{tts_end_time - tts_start_time:.3f}秒")

    except KeyboardInterrupt:
        print("\n程序已结束")
        break
    except Exception as e:
        logger.error(f"主循环异常: {str(e)}")