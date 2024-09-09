import pyaudio
import wave
import os
import random
import gradio as gr
import time
import torch
import gc
import warnings
warnings.filterwarnings('ignore')
from zhconv import convert
from LLM import LLM
from TTS import EdgeTTS
from src.cost_time import calculate_time

from configs import *

default_system = '你是一个很有帮助的助手'
#录制音频
voice = 'zh-CN-XiaoxiaoNeural'
# 语音合成的方法
tts_method == 'Edge-TTS'
rate = 0
# 设置默认的prompt
prefix_prompt = '''请用少于25个字回答以下问题\n\n'''

edgetts = EdgeTTS()

# 设定默认参数值，可修改
blink_every = True
size_of_image = 256
preprocess_type = 'crop'
facerender = 'facevid2vid'
enhancer = False
is_still_mode = False
exp_weight = 1
use_ref_video = False
ref_video = None
ref_info = 'pose'
use_idle_mode = False
length_of_audio = 5
# 音量
volume = 0
pitch = 0

os.environ["GRADIO_TEMP_DIR"]= './temp'
os.environ["WEBUI"] = "true"

def record():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 3
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

# 语音转文字
def Asr(audio):
    try:
        question = asr.transcribe(audio)
        question = convert(question, 'zh-cn')
    except Exception as e:
        print("ASR Error: ", e)
        question = 'Gradio存在一些bug，麦克风模式有时候可能音频还未传入，请重新点击一下语音识别即可'
        gr.Warning(question)
    return question


def Talker_response_img():
    record()
    # 用 FunASR模型进行语音识别
    asr = FunASR()
    text = asr.transcribe(output.wav)
    # 用Qwen模型进行回复
    llm_class = LLM(mode='offline')
    llm = llm_class.init_model('Qwen')
    answer = llm.generate(text, default_system)
    print(answer)
    # 将结果进行语音合成
    driven_audio, driven_vtt = TTS_response(answer, voice, rate, volume, pitch)
    # 进行图像的驱动
    from TFG import ERNeRF
    nerf = ERNeRF()
    nerf.init_model('./checkpoints/Obama_ave.pth', './checkpoints/Obama.json')
    nerf.predict(driven_audio)
    video = talker.predict(driven_audio)
    return video

def image():
    with gr.Tabs(elem_id="sadtalker_genearted"):
        gen_video = gr.Video(label="数字人视频", format="mp4")
    submit = gr.Button('🎬 生成数字人视频', elem_id="sadtalker_generate", variant='primary')
    submit.click(
        fn=Talker_response_img,
        outputs=[gen_video]

    )
def TTS_response(answer, voice, rate, volume, pitch):
    edgetts.predict(text, voice, rate, volume, pitch, 'answer.wav', 'answer.vtt')
    return 'answer.wav', 'answer.vtt'

if __name__ == "__main__":
    image()
