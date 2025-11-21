from faster_whisper import WhisperModel
import ffmpeg
import os
from datetime import datetime

# 模型选择配置
# 可选模型: "tiny", "base", "small", "medium", "large-v3", "turbo"
# 精度排序: tiny < base < small < medium < large-v3 ≈ turbo
# 速度排序: tiny > base > small > medium > turbo > large-v3

# 推荐配置：
# - 如果追求速度: "small"
# - 如果追求精度: "medium" 或 "large-v3"
# - 如果要平衡: "turbo" (最新优化版本)

# os.environ['HTTP_PROXY'] = 'http://..:.7@10.10.10.3:7890'
# os.environ['HTTPS_PROXY'] = 'http://..:.7@10.10.10.3:7890'
# transfer.xethub.hf.co
MODEL_SIZE = "medium"  # 🔥tiny, base, small, medium, large-v3
COMPUTE_TYPE = "int8"  # 可选: "int8", "float16", "float32"

print(f"使用模型: {MODEL_SIZE}")
print(f"计算类型: {COMPUTE_TYPE}")


# Step 1: Convert MP3/WAV to 16kHz mono wav
def convert_to_wav16k(input_path, output_path="converted.wav"):
    if os.path.exists(output_path):
        print(f"WAV文件 '{output_path}' 已存在，跳过转换步骤")
        return output_path

    print(f"正在转换 '{input_path}' 到 '{output_path}'...")
    (
        ffmpeg.input(input_path)
        .output(output_path, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    print(f"转换完成: {output_path}")
    return output_path


# Step 2: Load model
print(f"正在加载模型 '{MODEL_SIZE}'...")
model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)  # device="cuda",
print("模型加载完成!")

name = '11369449831'  # '11389271598' ,11389284145','11389274136'
# Step 3: 转换音频
wav_path = convert_to_wav16k(f"../data/{name}.mp3", f"data/{name}_converted.wav")  # 或 your_long_audio.wav

# Step 4: 创建文本文件并写入头部信息
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"transcription_{MODEL_SIZE}_{name}_{timestamp}.txt"

print(f"正在创建文本文件: {output_filename}")
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(f"音频文件: {wav_path}\n")
    f.write(f"使用模型: {MODEL_SIZE}\n")
    f.write(f"计算类型: {COMPUTE_TYPE}\n")
    f.write(f"识别开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("-" * 50 + "\n\n")

# Step 5: 识别并逐句输出和写入
segments, info = model.transcribe(wav_path, beam_size=5, language="zh", word_timestamps=False)

print("开始识别...")
print(f"实时保存到: {output_filename}")

with open(output_filename, "a", encoding="utf-8") as f:
    for segment in segments:
        segment_text = f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}"
        print(segment_text)
        f.write(segment_text + "\n")
        f.flush()  # 确保立即写入磁盘

    # 在最后添加完成信息
    f.write(f"\n" + "-" * 50 + "\n")
    f.write(f"识别完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"语言: {info.language}\n")
    f.write(f"总时长: {info.duration:.1f}秒\n")

print(f"\n识别完成！完整结果保存在: {output_filename}")
