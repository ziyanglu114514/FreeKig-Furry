import requests
import subprocess
import sounddevice as sd
import numpy as np
import threading
import collections

class TTSStreamingClient:
    def __init__(self,
                 api_url="http://127.0.0.1:9880/",
                 sample_rate=24000,
                 channels=1,
                 chunk_size=128,
                 dtype="int16",
                 on_audio_start=None):   # 新增
        self.api_url = api_url
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.pcm_queue = collections.deque()
        self.play_thread = None
        self.reader_thread = None
        self.ffmpeg = None
        self.on_audio_start = on_audio_start   # 新增
        self._audio_started = False            # 新增
        # 计算2秒对应的字节数
        self.bytes_to_skip = int(0.3 * sample_rate * channels * 2)  # 2秒 * 采样率 * 通道数 * 每个样本的字节数(int16=2字节)

    def tts_stream(self, text, text_language="zh"):
        headers = {"Content-Type": "application/json"}
        payload = {
            "text": text,
            "text_language": text_language,
            "cut_punc": "，。",
        }
        resp = requests.post(self.api_url, json=payload, stream=True)
        resp.raise_for_status()
        return resp.iter_content(chunk_size=self.chunk_size)

    def play_pcm_stream(self):
        # 播放线程：不断从队列取音频块
        with sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.chunk_size,
            latency='low'
        ) as stream:
            while True:
                chunk = self.pcm_queue.popleft() if self.pcm_queue else None
                if chunk is None:
                    sd.sleep(5)
                    continue
                if chunk == b"__end__":
                    break
                # 关键：第一次开始播放时打点
                if not self._audio_started:
                    self._audio_started = True
                    if self.on_audio_start:
                        self.on_audio_start()
                stream.write(chunk)

    def ffmpeg_reader(self):
        buffer = bytearray()  # 用于累积数据的缓冲区
        skip_remaining = self.bytes_to_skip  # 剩余需要跳过的字节数
        
        while True:
            chunk = self.ffmpeg.stdout.read(self.chunk_size)
            if not chunk:
                break
                
            # 如果还需要跳过数据
            if skip_remaining > 0:
                # 将当前块添加到缓冲区
                buffer.extend(chunk)
                
                # 如果缓冲区数据足够跳过
                if len(buffer) >= skip_remaining:
                    # 计算剩余有效数据
                    remaining_data = buffer[skip_remaining:]
                    
                    # 将有效数据放入队列（如果有）
                    if remaining_data:
                        self.pcm_queue.append(bytes(remaining_data))
                    
                    # 重置跳过状态
                    skip_remaining = 0
                    buffer = bytearray()  # 清空缓冲区
                # 否则继续跳过
                continue
            
            # 不需要跳过时直接放入队列
            self.pcm_queue.append(chunk)
        
        # 处理流结束时的剩余数据
        if skip_remaining <= 0 and buffer:
            self.pcm_queue.append(bytes(buffer))
            
        self.pcm_queue.append(b"__end__")

    def stream_tts(self, text, text_language="zh"):
        # 启动播放线程
        self.play_thread = threading.Thread(target=self.play_pcm_stream)
        self.play_thread.start()

        # 启动 ffmpeg
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel", "quiet",
                "-i", "pipe:0",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", str(self.channels),
                "pipe:1"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=self.chunk_size * 16
        )

        # 启动 ffmpeg PCM 读线程
        self.reader_thread = threading.Thread(target=self.ffmpeg_reader)
        self.reader_thread.start()

        # 推送 ogg 流到 ffmpeg
        try:
            for ogg_chunk in self.tts_stream(text, text_language):
                if ogg_chunk:
                    self.ffmpeg.stdin.write(ogg_chunk)
            self.ffmpeg.stdin.close()
        finally:
            self.reader_thread.join()
            self.play_thread.join()
            self.ffmpeg.wait()

if __name__ == "__main__":
    # 示例用法
    import time

    # 定义回调
    tts_real_start_time = None  # 全局变量用于保存开始播放时间

    def on_audio_start():
        global tts_real_start_time
        tts_real_start_time = time.time()
        print(f"【DEBUG】音频流真正开始播放: {tts_real_start_time}")

    # 实例化客户端
    client = TTSStreamingClient(
        api_url="http://127.0.0.1:9880/",
        sample_rate=24000,
        channels=1,
        chunk_size=128,
        on_audio_start=on_audio_start    # 传入回调
    )

    # 在主流程，统计从人说完话到tts_real_start_time的时间
    # 假设 speech_end_time = time.time() 记录在主流程
    # 调用 client.stream_tts 后（它阻塞到播放完），用 tts_real_start_time-speech_end_time 即可
    print("start")
    # 示范：
    speech_end_time = time.time()
    tts_real_start_time = None

    client.stream_tts("你好，测试一下流式延迟", text_language="zh")

    if tts_real_start_time:
        elapsed = tts_real_start_time - speech_end_time
        print(f"从人说完话到音频真正播放的延迟: {elapsed:.3f} 秒")
    else:
        print("未检测到音频播放起始。")